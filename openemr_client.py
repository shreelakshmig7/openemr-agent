"""
openemr_client.py
-----------------
AgentForge — Healthcare RCM AI Agent — OpenEMR FHIR R4 Client
--------------------------------------------------------------
Async FHIR R4 client for the OpenEMR instance running inside Docker.

Networking (docker/development-easy/docker-compose.yml):
  - openemr service: ports 9300:443  →  HTTPS at https://localhost:9300
  - OE_USER=admin, OE_PASS=pass
  - OPENEMR_SETTING_oauth_password_grant=3  (password grant enabled)
  - OPENEMR_SETTING_rest_fhir_api=1         (FHIR API enabled)
  - OPENEMR_SETTING_site_addr_oath='https://localhost:9300'

OAuth2 flow:
  1. If no client_id is configured, the client auto-registers via
     POST /oauth2/default/registration (dynamic client registration).
  2. POST /oauth2/default/token  with grant_type=password to obtain
     a bearer token; the token is cached and reused until it expires
     (refreshed automatically 30 s before expiry).
  3. All FHIR requests carry the bearer token in the Authorization header.

Usage (async context manager — preferred):
    async with OpenEMRClient() as client:
        bundle = await client.get_patients(family="Smith")
        obs    = await client.post_observation({...})

Usage (manual lifecycle):
    client = OpenEMRClient()
    await client.connect()
    bundle = await client.get_patients()
    await client.close()

Author: Shreelakshmi Gopinatha Rao
Project: AgentForge — Healthcare RCM AI Agent
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FHIR scopes required for Patient/MedicationRequest/Observation read+write
# ---------------------------------------------------------------------------
_FHIR_SCOPES = (
    "openid "
    # FHIR R4 API
    "api:fhir "
    "user/Patient.read "
    "user/Patient.write "
    "user/MedicationRequest.read "
    "user/MedicationRequest.write "
    "user/Observation.read "
    "user/Observation.write "
    "user/AllergyIntolerance.read "
    "patient/AllergyIntolerance.read "
    # Standard REST API (OpenEMR dot-notation scopes: user/{resource}.{action})
    "api:oemr "
    "user/encounter.c "      # create encounters  (POST /api/patient/:uuid/encounter)
    "user/encounter.r "      # read encounters
    "user/encounter.write "  # encounter write (broad)
    "user/vital.c "          # create vitals
    "user/vital.r "          # read vitals
    "user/vital.write "      # vitals write (broad)
    "user/soap_note.write "     # SOAP note write — required for POST …/soap_note
    "user/soap_note.read "      # SOAP note read
    "user/soap_note.crus "      # SOAP note full CRUD (create/read/update/search)
    "user/allergy.write "       # allergy write — POST /api/patient/{uuid}/allergy
    "user/allergy.read "        # allergy read
    "user/allergy.cruds "       # allergy full CRUD
    "user/medical_problem.write " # medical problem write
    "user/medical_problem.read "  # medical problem read
    "user/medical_problem.cruds " # medical problem full CRUD
    "user/insurance.write "     # insurance write — POST /api/patient/{uuid}/insurance
    "user/insurance.read "      # insurance read
    "user/insurance.crus"       # insurance full CRUD
)

# Seconds before token expiry at which a proactive refresh is triggered.
_TOKEN_REFRESH_BUFFER_S = 30


class OpenEMRAuthError(Exception):
    """Raised when OAuth2 registration or token acquisition fails."""


class OpenEMRAPIError(Exception):
    """Raised when a FHIR API call returns a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"OpenEMR FHIR API error {status_code}: {body}")


class OpenEMRClient:
    """
    Async OpenEMR FHIR R4 client with OAuth2 Password Grant authentication.

    All network I/O is performed via ``httpx.AsyncClient`` with ``verify=False``
    to accommodate the self-signed TLS certificate on the Docker container.

    Args:
        base_url:      HTTPS base URL of the OpenEMR container.
                       Defaults to ``OPENEMR_BASE_URL`` env var, then
                       ``https://localhost:9300``.
        username:      OpenEMR admin username (OE_USER in docker-compose).
        password:      OpenEMR admin password (OE_PASS in docker-compose).
        client_id:     Pre-registered OAuth2 client ID.  If *None*, the
                       client auto-registers on first use and caches the
                       result in memory.
        client_secret: OAuth2 client secret (required for confidential
                       clients; may be empty for public clients).
        site:          OpenEMR site identifier; default is ``"default"``.
        timeout:       HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: str = "admin",
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        site: str = "default",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("OPENEMR_BASE_URL", "https://localhost:9300")
        ).rstrip("/")
        self.username = os.getenv("OPENEMR_USERNAME", username)
        self.password = password or os.getenv("OPENEMR_PASSWORD", "pass")
        self.site = site
        self.timeout = timeout

        # OAuth2 credentials — may be resolved lazily via auto-registration.
        self._client_id: Optional[str] = client_id or os.getenv("OPENEMR_CLIENT_ID")
        self._client_secret: Optional[str] = client_secret or os.getenv(
            "OPENEMR_CLIENT_SECRET", ""
        )

        # Token state
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

        # Underlying HTTP transport (initialised in connect / __aenter__)
        self._http: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the underlying HTTP connection pool."""
        if self._http is None:
            self._http = httpx.AsyncClient(
                verify=False,   # ignore self-signed cert on Docker container
                timeout=self.timeout,
            )
            logger.debug("OpenEMRClient: HTTP transport initialised (verify=False).")

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
            logger.debug("OpenEMRClient: HTTP transport closed.")

    async def __aenter__(self) -> "OpenEMRClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── URL helpers ──────────────────────────────────────────────────────────

    @property
    def _token_url(self) -> str:
        return f"{self.base_url}/oauth2/{self.site}/token"

    @property
    def _registration_url(self) -> str:
        return f"{self.base_url}/oauth2/{self.site}/registration"

    @property
    def _fhir_base(self) -> str:
        return f"{self.base_url}/apis/{self.site}/fhir"

    @property
    def _api_base(self) -> str:
        """Base URL for the standard (non-FHIR) OpenEMR REST API."""
        return f"{self.base_url}/apis/{self.site}/api"

    # ── OAuth2 helpers ───────────────────────────────────────────────────────

    async def _register_client(self) -> tuple[str, str]:
        """
        Dynamically register a confidential OAuth2 client with OpenEMR.

        OpenEMR implements RFC 7591 (OAuth 2.0 Dynamic Client Registration).
        The returned ``client_id`` / ``client_secret`` are stored on the
        instance and reused for the lifetime of this client object.

        Returns:
            (client_id, client_secret)

        Raises:
            OpenEMRAuthError: if registration fails.
        """
        payload = {
            "application_type": "private",
            "redirect_uris": [f"{self.base_url}/callback"],
            "client_name": "AgentForge-RCM",
            "token_endpoint_auth_method": "client_secret_post",
            "contacts": ["admin@agentforge.local"],
            "scope": _FHIR_SCOPES,
        }
        try:
            resp = await self._http.post(self._registration_url, json=payload)
        except httpx.HTTPError as exc:
            raise OpenEMRAuthError(
                f"Client registration request failed: {exc}"
            ) from exc

        if resp.status_code not in (200, 201):
            raise OpenEMRAuthError(
                f"Client registration returned {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        client_id: str = data["client_id"]
        client_secret: str = data.get("client_secret", "")
        logger.info(
            "OpenEMRClient: dynamic client registered (client_id=%s).", client_id
        )
        return client_id, client_secret

    def _activate_client_via_docker(self, client_id: str) -> bool:
        """
        Activate a newly registered OAuth2 client by running a MySQL UPDATE
        inside the OpenEMR Docker container.

        OpenEMR's dynamic client registration creates clients with
        ``is_enabled = 0`` (pending admin approval).  In a local Docker dev
        environment this method provides an automated activation path so the
        seed script can run without manual admin-UI intervention.

        The container is located by its published port (9300) so that the
        method stays portable regardless of container name or compose project.

        The MySQL credentials match the ``docker-compose.yml`` values:
            MYSQL_USER=openemr  MYSQL_PASS=openemr  MYSQL_HOST=mysql

        Args:
            client_id: The ``client_id`` returned by ``_register_client()``.

        Returns:
            ``True`` if the activation SQL executed successfully,
            ``False`` on any error (docker not found, container absent, etc.).
        """
        import subprocess  # stdlib — import here to keep top-level imports lean

        # Sanitise client_id: these are server-generated opaque strings, but
        # guard against any unexpected characters before interpolating into SQL.
        safe_id = client_id.replace("'", "\\'")
        sql = (
            f"UPDATE oauth_clients SET is_enabled=1 "
            f"WHERE client_id='{safe_id}' AND is_enabled=0;"
        )

        try:
            # Find the container that has port 9300 exposed (openemr service).
            ps = subprocess.run(
                ["docker", "ps", "-q", "--filter", "publish=9300"],
                capture_output=True, text=True, timeout=5,
            )
            containers = [c.strip() for c in ps.stdout.splitlines() if c.strip()]
            if not containers:
                logger.warning(
                    "OpenEMRClient: no Docker container found on port 9300 — "
                    "cannot auto-activate client %s.",
                    client_id,
                )
                return False

            container = containers[0]
            result = subprocess.run(
                [
                    "docker", "exec", container,
                    "mysql",
                    "-u", "openemr",
                    "-popenemr",          # password from docker-compose MYSQL_PASS
                    "-h", "mysql",        # service name from docker-compose
                    "openemr",            # database name
                    "-e", sql,
                ],
                capture_output=True, text=True, timeout=10,
            )

            if result.returncode == 0:
                logger.info(
                    "OpenEMRClient: client_id=%s activated via docker exec "
                    "(container=%s).",
                    client_id, container,
                )
                return True

            logger.warning(
                "OpenEMRClient: docker exec activation failed "
                "(rc=%d, stderr=%s).",
                result.returncode, result.stderr.strip(),
            )
            return False

        except FileNotFoundError:
            logger.warning(
                "OpenEMRClient: docker CLI not found — cannot auto-activate "
                "client %s. Approve it manually at %s/interface/smart/"
                "register-app.php",
                client_id, self.base_url,
            )
            return False
        except Exception as exc:
            logger.warning(
                "OpenEMRClient: docker exec activation raised %s: %s",
                type(exc).__name__, exc,
            )
            return False

    async def _fetch_token(self) -> None:
        """
        Obtain a new bearer token via OAuth2 Password Grant and cache it.

        Automatically registers and activates a client first if ``client_id``
        is not set.  Activation is attempted via ``docker exec`` because
        OpenEMR creates dynamically-registered clients with ``is_enabled=0``
        (pending admin approval) — the Docker MySQL call flips the flag to 1
        so the token endpoint accepts the client immediately.

        Raises:
            OpenEMRAuthError: if the token request fails.
        """
        if not self._client_id:
            self._client_id, self._client_secret = await self._register_client()
            # OpenEMR registers clients as disabled (is_enabled=0) by default.
            # Activate it now so the token request below succeeds without
            # requiring a manual admin-UI approval step.
            activated = self._activate_client_via_docker(self._client_id)
            if not activated:
                logger.warning(
                    "OpenEMRClient: auto-activation unavailable for client_id=%s. "
                    "If the token request fails, approve the client at: "
                    "%s/interface/smart/register-app.php  "
                    "or set OPENEMR_CLIENT_ID / OPENEMR_CLIENT_SECRET in .env.",
                    self._client_id, self.base_url,
                )

        form_data: dict[str, str] = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "user_role": "users",    # OpenEMR extension: "users" for admin/provider, "patient" for portal
            "client_id": self._client_id,
            "scope": _FHIR_SCOPES,
        }
        if self._client_secret:
            form_data["client_secret"] = self._client_secret

        try:
            resp = await self._http.post(
                self._token_url,
                data=form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise OpenEMRAuthError(
                f"Token request failed: {exc}"
            ) from exc

        if resp.status_code != 200:
            body = resp.text
            # Provide an actionable hint when the client is still not approved.
            if "invalid_client" in body:
                raise OpenEMRAuthError(
                    f"Token endpoint returned {resp.status_code}: {body}\n\n"
                    f"  The OAuth2 client '{self._client_id}' is not yet enabled.\n"
                    f"  Approve it at: {self.base_url}/interface/smart/register-app.php\n"
                    f"  Or set OPENEMR_CLIENT_ID / OPENEMR_CLIENT_SECRET in .env and skip "
                    f"dynamic registration."
                )
            raise OpenEMRAuthError(
                f"Token endpoint returned {resp.status_code}: {body}"
            )

        token_data = resp.json()
        self._access_token = token_data["access_token"]
        expires_in: int = int(token_data.get("expires_in", 3600))
        self._token_expires_at = time.monotonic() + expires_in

        logger.info(
            "OpenEMRClient: bearer token obtained (expires_in=%ds).", expires_in
        )

    async def _ensure_token(self) -> str:
        """
        Return a valid bearer token, fetching or refreshing when necessary.

        A proactive refresh is triggered ``_TOKEN_REFRESH_BUFFER_S`` seconds
        before the cached token would expire.
        """
        if (
            self._access_token
            and time.monotonic() < self._token_expires_at - _TOKEN_REFRESH_BUFFER_S
        ):
            return self._access_token

        logger.debug(
            "OpenEMRClient: %s — fetching new token.",
            "token expired or absent" if not self._access_token else "proactive refresh",
        )
        await self._fetch_token()
        return self._access_token  # type: ignore[return-value]

    async def _auth_headers(self) -> dict[str, str]:
        """Return HTTP headers with a valid bearer token and FHIR content types."""
        token = await self._ensure_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }

    # ── Internal request helper ──────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, str]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute an authenticated FHIR R4 request and return the parsed JSON body.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, …).
            path:   Path relative to the FHIR base URL (e.g. ``"/Patient"``).
            params: URL query parameters.
            json:   Request body (serialised as JSON).

        Returns:
            Parsed JSON response body.

        Raises:
            RuntimeError:     if ``connect()`` / ``__aenter__`` was not called.
            OpenEMRAPIError:  if the server returns a non-2xx status code.
        """
        if self._http is None:
            raise RuntimeError(
                "OpenEMRClient is not connected. "
                "Use 'async with OpenEMRClient() as client:' or call connect() first."
            )

        url = f"{self._fhir_base}{path}"
        headers = await self._auth_headers()

        try:
            resp = await self._http.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json,
            )
        except httpx.HTTPError as exc:
            raise OpenEMRAPIError(0, str(exc)) from exc

        if resp.status_code not in range(200, 300):
            raise OpenEMRAPIError(resp.status_code, resp.text)

        body: dict[str, Any] = resp.json() if resp.content else {}

        # OpenEMR's FHIR write endpoints (201 Created) return a non-standard
        # body: {"pid": <int>, "uuid": "<uuid>"} instead of the full FHIR
        # resource.  The "uuid" IS the FHIR resource id used by subsequent
        # GET /Patient/{id} calls.  Normalise to "id" so all callers can use
        # response["id"] uniformly.
        if not body.get("id") and body.get("uuid"):
            body["id"] = body["uuid"]

        return body

    # ── FHIR R4 Public API ───────────────────────────────────────────────────

    async def get_patients(self, **search_params: str) -> dict[str, Any]:
        """
        Search for Patient resources  →  ``GET /apis/default/fhir/Patient``.

        Keyword arguments are forwarded directly as FHIR search parameters.
        See https://www.hl7.org/fhir/R4/patient.html#search for the full list.

        Common parameters:
            family (str):      Family (last) name.
            given (str):       Given (first) name.
            birthdate (str):   ISO-8601 date, e.g. ``"1980-01-15"``.
            identifier (str):  MRN or other identifier token.
            _count (str):      Maximum results per page, e.g. ``"10"``.

        Returns:
            A FHIR ``Bundle`` resource (dict) containing matched ``Patient``
            entries.  Returns an empty Bundle when no patients match.

        Example::

            bundle = await client.get_patients(family="Smith", given="John")
            for entry in bundle.get("entry", []):
                patient = entry["resource"]
                print(patient["id"], patient.get("birthDate"))
        """
        logger.debug(
            "OpenEMRClient: GET /Patient params=%s",
            search_params or "<all>",
        )
        return await self._request(
            "GET",
            "/Patient",
            params=search_params if search_params else None,
        )

    async def post_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Create an Observation resource  →  ``POST /apis/default/fhir/Observation``.

        Args:
            observation: A FHIR R4 ``Observation`` resource.  Required fields:

                * ``resourceType`` — must be ``"Observation"``
                * ``status``       — e.g. ``"final"``, ``"preliminary"``
                * ``code``         — ``{ "coding": [{ "system", "code", "display" }] }``
                  Use LOINC codes per project standards (`.cursorrules`).
                * ``subject``      — ``{ "reference": "Patient/{id}" }``

        Returns:
            The newly created ``Observation`` resource with the server-assigned
            ``id`` and ``meta.versionId``.

        Raises:
            ValueError:       if ``resourceType`` is not ``"Observation"``.
            OpenEMRAPIError:  if the server rejects the resource.

        Example::

            obs = await client.post_observation({
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code":   "8867-4",
                        "display": "Heart rate",
                    }]
                },
                "subject": {"reference": "Patient/1"},
                "valueQuantity": {
                    "value": 72, "unit": "beats/min",
                    "system": "http://unitsofmeasure.org", "code": "/min",
                },
            })
            print("Created Observation:", obs["id"])
        """
        resource_type = observation.get("resourceType")
        if resource_type != "Observation":
            raise ValueError(
                f"Expected resourceType 'Observation', got '{resource_type}'. "
                "Pass a valid FHIR R4 Observation resource dict."
            )

        logger.debug(
            "OpenEMRClient: POST /Observation (subject=%s, code=%s)",
            observation.get("subject", {}).get("reference", "?"),
            observation.get("code", {})
            .get("coding", [{}])[0]
            .get("code", "?"),
        )
        return await self._request("POST", "/Observation", json=observation)

    async def post_patient(self, patient: dict[str, Any]) -> dict[str, Any]:
        """
        Create a Patient resource  →  ``POST /apis/default/fhir/Patient``.

        Args:
            patient: A FHIR R4 ``Patient`` resource dict.  Required fields:

                * ``resourceType`` — must be ``"Patient"``
                * ``name``         — list with at least one ``HumanName`` entry
                * ``gender``       — ``"male"`` | ``"female"`` | ``"other"`` | ``"unknown"``
                * ``birthDate``    — ISO-8601 date string (``"YYYY-MM-DD"``)

        Returns:
            The newly created ``Patient`` resource with the server-assigned ``id``.

        Raises:
            ValueError:       if ``resourceType`` is not ``"Patient"``.
            OpenEMRAPIError:  if the server rejects the resource.

        Example::

            patient = await client.post_patient({
                "resourceType": "Patient",
                "name": [{"use": "official", "family": "Smith", "given": ["John"]}],
                "gender": "male",
                "birthDate": "1965-03-15",
            })
            print("Created Patient:", patient["id"])
        """
        resource_type = patient.get("resourceType")
        if resource_type != "Patient":
            raise ValueError(
                f"Expected resourceType 'Patient', got '{resource_type}'. "
                "Pass a valid FHIR R4 Patient resource dict."
            )

        name_entry = patient.get("name", [{}])[0] if patient.get("name") else {}
        logger.debug(
            "OpenEMRClient: POST /Patient (family=%s, given=%s)",
            name_entry.get("family", "?"),
            " ".join(name_entry.get("given", ["?"])),
        )
        return await self._request("POST", "/Patient", json=patient)

    async def post_medication_request(
        self, medication_request: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create a MedicationRequest resource
        →  ``POST /apis/default/fhir/MedicationRequest``.

        Args:
            medication_request: A FHIR R4 ``MedicationRequest`` resource dict.
                Required fields:

                * ``resourceType``              — must be ``"MedicationRequest"``
                * ``status``                    — e.g. ``"active"``
                * ``intent``                    — e.g. ``"order"``
                * ``medicationCodeableConcept`` — ``{ "text": "DrugName" }``
                * ``subject``                   — ``{ "reference": "Patient/{id}" }``

        Returns:
            The newly created ``MedicationRequest`` resource with the
            server-assigned ``id``.

        Raises:
            ValueError:       if ``resourceType`` is not ``"MedicationRequest"``.
            OpenEMRAPIError:  if the server rejects the resource.

        Example::

            med = await client.post_medication_request({
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {"text": "Metformin"},
                "subject": {"reference": "Patient/1"},
                "dosageInstruction": [{"text": "500mg twice daily"}],
            })
            print("Created MedicationRequest:", med["id"])
        """
        resource_type = medication_request.get("resourceType")
        if resource_type != "MedicationRequest":
            raise ValueError(
                f"Expected resourceType 'MedicationRequest', got '{resource_type}'. "
                "Pass a valid FHIR R4 MedicationRequest resource dict."
            )

        med_name = (
            medication_request.get("medicationCodeableConcept", {}).get("text", "?")
        )
        subject = medication_request.get("subject", {}).get("reference", "?")
        logger.debug(
            "OpenEMRClient: POST /MedicationRequest (medication=%s, subject=%s)",
            med_name,
            subject,
        )
        return await self._request("POST", "/MedicationRequest", json=medication_request)

    async def get_fhir_medications(self, patient_uuid: str) -> list[dict[str, Any]]:
        """
        Fetch active medications for a patient via FHIR R4 MedicationRequest.
        →  ``GET /apis/default/fhir/MedicationRequest?patient={uuid}&status=active``

        Parses each MedicationRequest entry into a simplified dict matching
        the mock_data/medications.json schema so downstream tools receive a
        consistent shape regardless of data source.

        Args:
            patient_uuid: FHIR Patient UUID.

        Returns:
            List of dicts: [{"name": str, "dose": str, "frequency": str}, ...]
            Empty list on error or no medications found.
        """
        try:
            data = await self._request(
                "GET",
                "/MedicationRequest",
                # Include all prescription statuses so medications seeded with
                # different active flags (active, stopped, on-hold, completed)
                # are all returned rather than silently dropped.
                params={"patient": patient_uuid, "_count": "50"},
            )
            meds: list[dict[str, Any]] = []
            for entry in data.get("entry", []):
                res = entry.get("resource", {})
                if res.get("resourceType") != "MedicationRequest":
                    continue

                # Medication name — prefer medicationCodeableConcept.text
                med_cc   = res.get("medicationCodeableConcept", {})
                name     = med_cc.get("text", "")
                if not name:
                    codings = med_cc.get("coding", [{}])
                    name    = codings[0].get("display", "") if codings else ""
                if not name:
                    name = res.get("medicationReference", {}).get("display", "Unknown")

                # Dose + frequency from dosageInstruction[0]
                di        = (res.get("dosageInstruction") or [{}])[0]
                dose_qty  = di.get("doseAndRate", [{}])
                dose_str  = ""
                if dose_qty:
                    dq = dose_qty[0].get("doseQuantity", {})
                    dose_str = f"{dq.get('value', '')} {dq.get('unit', '')}".strip()
                freq_str = di.get("text", "") or di.get("patientInstruction", "")

                if name:
                    meds.append({
                        "name":      name,
                        "dose":      dose_str  or "per prescription",
                        "frequency": freq_str  or "per prescription",
                    })
            return meds
        except Exception as exc:
            logger.warning("OpenEMRClient.get_fhir_medications: %s", exc)
            return []

    async def get_fhir_allergies(self, patient_uuid: str) -> list[str]:
        """
        Fetch documented allergies for a patient via FHIR R4 AllergyIntolerance.
        →  ``GET /apis/default/fhir/AllergyIntolerance?patient={uuid}``

        Returns a flat list of allergen names (strings) matching the format
        used in mock_data/patients.json so downstream tools are consistent.

        Args:
            patient_uuid: FHIR Patient UUID.

        Returns:
            List of allergen name strings, e.g. ["Penicillin", "Sulfa"].
            Empty list on error or no allergies found.
        """
        try:
            data = await self._request(
                "GET",
                "/AllergyIntolerance",
                params={"patient": patient_uuid, "_count": "50"},
            )
            allergens: list[str] = []
            for entry in data.get("entry", []):
                res = entry.get("resource", {})
                if res.get("resourceType") != "AllergyIntolerance":
                    continue
                code  = res.get("code", {})
                name  = code.get("text", "")
                if not name:
                    codings = code.get("coding", [{}])
                    name    = codings[0].get("display", "") if codings else ""
                if name:
                    allergens.append(name)
            return allergens
        except Exception as exc:
            logger.warning("OpenEMRClient.get_fhir_allergies: %s", exc)
            return []

    # ── Standard REST API (non-FHIR) ────────────────────────────────────────

    async def post_encounter(
        self,
        patient_uuid: str,
        *,
        date: Optional[str] = None,
        reason: str = "AgentForge clinical data sync",
        pc_catid: str = "5",
    ) -> dict[str, Any]:
        """
        Create a patient Encounter via the standard OpenEMR REST API
        →  ``POST /apis/default/api/patient/:puuid/encounter``.

        This endpoint is used when FHIR Observation write is not available in
        the target OpenEMR version.  Creating an encounter proves a successful
        round-trip to OpenEMR and serves as the sync anchor for evidence staging
        rows (the returned encounter UUID is stored as ``fhir_observation_id``).

        Args:
            patient_uuid: FHIR / internal UUID of the target patient.
            date:         Encounter date (ISO-8601, ``"YYYY-MM-DD"``).
                          Defaults to today in UTC.
            reason:       Free-text reason for the encounter.
            pc_catid:     OpenEMR encounter category ID (default ``"5"`` = other).

        Returns:
            Dict with at least ``"id"`` (encounter UUID) and ``"encounter"``
            (internal integer encounter ID).

        Raises:
            RuntimeError:     if the client is not connected.
            OpenEMRAPIError:  if OpenEMR rejects the request.

        Example::

            enc = await client.post_encounter(patient_uuid="a1312c03-...")
            print("Encounter UUID:", enc["id"])
        """
        if self._http is None:
            raise RuntimeError(
                "OpenEMRClient is not connected. "
                "Use 'async with OpenEMRClient() as client:' or call connect() first."
            )

        from datetime import date as _date_cls
        encounter_date = date or _date_cls.today().isoformat()

        url = f"{self._api_base}/patient/{patient_uuid}/encounter"
        token = await self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {
            "date":       encounter_date,
            "reason":     reason,
            "pc_catid":   pc_catid,
            "class_code": "AMB",    # required: AMB = ambulatory (FHIR ActCode)
        }

        logger.debug(
            "OpenEMRClient: POST /api/patient/%s/encounter (date=%s)",
            patient_uuid,
            encounter_date,
        )

        try:
            resp = await self._http.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise OpenEMRAPIError(0, str(exc)) from exc

        if resp.status_code not in range(200, 300):
            raise OpenEMRAPIError(resp.status_code, resp.text)

        body: dict[str, Any] = resp.json() if resp.content else {}
        data: dict[str, Any] = body.get("data") or body

        # Normalise UUID fields → "id" so callers use response["id"] uniformly.
        # OpenEMR returns "euuid" for encounter UUID (not the generic "uuid").
        if not data.get("id"):
            data["id"] = data.get("euuid") or data.get("uuid") or ""

        return data

    async def post_soap_note(
        self,
        patient_uuid: str,
        encounter_uuid: str,
        *,
        subjective: str = "",
        objective: str = "",
        assessment: str = "",
        plan: str = "",
    ) -> dict[str, Any]:
        """
        Create a SOAP Note under a patient encounter via the standard OpenEMR
        REST API → ``POST /apis/default/api/patient/:puuid/encounter/:euuid/soap_note``.

        This is the primary write path for clinical biomarker data when FHIR
        Observation writes are unavailable (confirmed blocked in the community
        demo build — see smart-configuration / scopes_supported).

        Args:
            patient_uuid:   FHIR / internal UUID of the target patient.
            encounter_uuid: UUID of the encounter under which to file the note.
            subjective:     S — patient-reported history, reason for visit.
            objective:      O — measured findings (lab values, biomarkers).
            assessment:     A — clinical interpretation.
            plan:           P — next steps / treatment decision.

        Returns:
            Dict with at least ``"id"`` (SOAP note UUID) from the response.

        Raises:
            RuntimeError:     if the client is not connected.
            OpenEMRAPIError:  if OpenEMR rejects the request.

        Example::

            note = await client.post_soap_note(
                patient_uuid="a1312c04-...",
                encounter_uuid="enc-uuid-...",
                subjective="Maria J. Gonzalez — Stage II invasive ductal carcinoma",
                objective="ER Positive (Allred 8/8) | PR Positive | HER2 Negative",
                assessment="HR+/HER2- — meets Aetna criteria for Palbociclib",
                plan="Proceed with Palbociclib 125mg per AgentForge RCM audit",
            )
        """
        if self._http is None:
            raise RuntimeError(
                "OpenEMRClient is not connected. "
                "Use 'async with OpenEMRClient() as client:' or call connect() first."
            )

        url = (
            f"{self._api_base}/patient/{patient_uuid}"
            f"/encounter/{encounter_uuid}/soap_note"
        )
        token = await self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, Any] = {
            "subjective": subjective,
            "objective":  objective,
            "assessment": assessment,
            "plan":       plan,
        }

        logger.debug(
            "OpenEMRClient: POST /api/patient/%s/encounter/%s/soap_note",
            patient_uuid,
            encounter_uuid,
        )

        try:
            resp = await self._http.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise OpenEMRAPIError(0, str(exc)) from exc

        if resp.status_code not in range(200, 300):
            raise OpenEMRAPIError(resp.status_code, resp.text)

        body: dict[str, Any] = resp.json() if resp.content else {}
        data: dict[str, Any] = body.get("data") or body

        if not data.get("id") and data.get("uuid"):
            data["id"] = data["uuid"]

        logger.info(
            "OpenEMRClient: SOAP note created (id=%s) for patient %s encounter %s.",
            data.get("id", "<unknown>"),
            patient_uuid,
            encounter_uuid,
        )
        return data

    async def post_bundle(
        self,
        bundle: dict[str, Any],
    ) -> dict[str, Any]:
        """
        POST each entry in a FHIR R4 Transaction Bundle individually.

        OpenEMR's FHIR layer does not register a ``POST /fhir/$transaction``
        route, so batch submission is not available.  This method iterates
        through ``bundle["entry"]``, derives the target URL from each entry's
        ``request.url`` field, and issues individual ``POST`` calls.

        Results are returned in the same order as ``bundle["entry"]``, so
        callers can correlate ``results[i]`` back to the original input fact
        at index *i*.

        Args:
            bundle: A FHIR R4 Transaction Bundle as produced by
                    ``fhir_mapper.map_to_bundle()``.

        Returns:
            dict with keys:

            * ``results``   — list of per-entry dicts (same order as bundle):
              - ``index``       (int)  position in bundle["entry"]
              - ``status``      ("success" | "failed")
              - ``fhir_id``     (str | None)  e.g. ``"Observation/42"``
              - ``http_status`` (int)  HTTP response code
              - ``error``       (str | None)  body snippet on failure
            * ``total``     — number of entries attempted
            * ``succeeded`` — count of successful POSTs
            * ``failed``    — count of failed POSTs

        Raises:
            RuntimeError: if the client is not connected.

        Example::

            bundle = fhir_mapper.map_to_bundle(patient_id, facts)
            result = await client.post_bundle(bundle)
            for r in result["results"]:
                if r["status"] == "success":
                    print("Created", r["fhir_id"])
        """
        if self._http is None:
            raise RuntimeError(
                "OpenEMRClient is not connected. "
                "Use 'async with OpenEMRClient() as client:' or call connect() first."
            )

        entries: list[dict[str, Any]] = bundle.get("entry", [])
        results: list[dict[str, Any]] = []

        for i, entry in enumerate(entries):
            resource: dict[str, Any] = entry.get("resource", {})
            resource_type: str       = resource.get("resourceType", "")
            request_info: dict       = entry.get("request", {})
            url_segment: str         = request_info.get("url", resource_type)
            method: str              = request_info.get("method", "POST").upper()

            if method != "POST":
                results.append({
                    "index":       i,
                    "status":      "failed",
                    "fhir_id":     None,
                    "http_status": 0,
                    "error":       f"Unsupported method '{method}' — only POST is handled.",
                })
                continue

            try:
                resp_body = await self._request("POST", f"/{url_segment}", json=resource)
                raw_id    = resp_body.get("id", "")
                fhir_id   = f"{resource_type}/{raw_id}" if raw_id else None
                results.append({
                    "index":       i,
                    "status":      "success",
                    "fhir_id":     fhir_id,
                    "http_status": 200,
                    "error":       None,
                })
                logger.debug(
                    "OpenEMRClient: post_bundle entry[%d] %s → %s.",
                    i, resource_type, fhir_id or "<no id>",
                )

            except OpenEMRAPIError as exc:
                results.append({
                    "index":       i,
                    "status":      "failed",
                    "fhir_id":     None,
                    "http_status": exc.status_code,
                    "error":       exc.body[:300],
                })
                logger.warning(
                    "OpenEMRClient: post_bundle entry[%d] %s failed — HTTP %d.",
                    i, resource_type, exc.status_code,
                )

        succeeded = sum(1 for r in results if r["status"] == "success")
        failed    = len(results) - succeeded
        logger.info(
            "OpenEMRClient: post_bundle complete — %d/%d succeeded, %d failed.",
            succeeded, len(results), failed,
        )
        return {
            "results":   results,
            "total":     len(results),
            "succeeded": succeeded,
            "failed":    failed,
        }
