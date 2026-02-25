**AgentForge**

**Healthcare Revenue Cycle Management AI Agent**

*Pre-Search Document*

Domain: Healthcare (OpenEMR/RCM) \| Framework: LangGraph Multi-Agent \|
LLM: Claude 3.5 Sonnet

February 2026

**Executive Summary**

This Pre-Search document defines the architecture for a production-ready
AI agent targeting Healthcare Revenue Cycle Management (RCM). The agent
extracts clinical data from provider notes and verifies whether the
documented evidence meets insurance coverage criteria --- directly
reducing claim denials and accelerating patient care authorization.

The agent uses a multi-agent LangGraph State Machine with a dedicated
Extractor Node and Auditor Node, a hybrid LLM cost strategy (Claude
Haiku for extraction, Claude 3.5 Sonnet for verification), and a
non-negotiable Evidence Attribution rule: no claim can be made without a
direct verbatim quote from the source document.

**Phase 1: Define Your Constraints**

**1.1 Domain Selection**

**Why This Decision:** *Revenue Cycle Management was chosen because it
sits at the intersection of clinical accuracy and financial consequence.
A wrong extraction does not just affect patient care --- it causes claim
denials, lost revenue, and delayed authorizations. This makes it one of
the highest-value problems in healthcare AI, where accuracy is
non-negotiable.*

**Final Decision: Healthcare Revenue Cycle Management --- extracting
clinical data from provider notes to verify insurance coverage
criteria.**

**Specific Use Cases**

-   Extract clinical evidence from provider notes to match against payer
    medical necessity criteria

-   Identify whether documented diagnoses (ICD-10) support the requested
    procedure (CPT codes)

-   Flag missing or contradictory clinical evidence before claim
    submission

-   Generate evidence-grounded summaries with direct source citations
    for human reviewers

**Data Sources**

  ---------------- --------------------------- ---------------------------
  **Data Source**  **Purpose**                 **Access Method**

  Scanned Clinical Provider notes, discharge   unstructured.io for
  PDFs             summaries, operative        extraction
                   reports                     

  HL7 FHIR API     Structured diagnoses,       FHIR R4 endpoints with
  (OpenEMR)        procedures, encounters      OAuth2

  Payer Policy     Insurance medical necessity Pinecone vector DB
  Databases        criteria (200+ page PDFs)   (chunked + indexed)

  ICD-10 / CPT     Validate diagnosis and      Custom Medical Fact-Checker
  Registry         procedure code accuracy     tool

  Mock Data        Development and automated   Local JSON + PDF fixtures
                   eval testing                
  ---------------- --------------------------- ---------------------------

**1.2 Scale & Performance**

**Why This Decision:** *In Revenue Cycle Management, accuracy is worth
more than speed. An insurance auditor would rather wait 30 seconds for a
perfectly cited extraction than receive a 5-second response that leads
to a denial. Denials trigger costly appeal processes and delay patient
care --- making accuracy the primary engineering constraint, not
latency.*

**Final Decision: Latency target: \< 30 seconds per document. Batch
throughput: 20--50 complex charts per hour.**

  ------------------ ---------------- -----------------------------------
  **Metric**         **Target**       **Rationale**

  Latency per        \< 30 seconds    Multi-agent reasoning over 50+ page
  document                            charts; accuracy over speed

  Batch throughput   20-50 charts per Realistic for a clinical
                     hour             authorization team

  Tool success rate  \> 95%           Per project requirements

  Eval pass rate     \> 80%           Per project requirements

  Hallucination rate \< 5%            Enforced by Evidence Attribution
                                      rule

  Verification       \> 90%           Per project requirements
  accuracy                            
  ------------------ ---------------- -----------------------------------

**Cost Projections**

  --------------- --------------- ------------------ --------------------------
  **Scale**       **Users**       **Charts/Month**   **Est. Monthly Cost**

  Pilot           100 users       \~5,000 charts     \~\$80--150/month

  Small clinic    1,000 users     \~50,000 charts    \~\$600--1,200/month

  Mid-size        10,000 users    \~500,000 charts   \~\$5,000--9,000/month
  hospital                                           

  Enterprise      100,000 users   \~5M charts        \~\$40,000--70,000/month
  --------------- --------------- ------------------ --------------------------

**1.3 Reliability Requirements**

**Cost of a Wrong Answer**

A wrong extraction causes two categories of harm: financial loss through
claim denials (each denial costs \$25--\$200 to rework) and clinical
risk through delayed care authorizations. This dual consequence makes
reliability the top engineering priority.

**Evidence Attribution Rule**

**Why This Decision:** *In healthcare AI, hallucination is not just a
quality issue --- it is a liability issue. If an agent claims a patient
has documented \'failed conservative therapy\' when the chart does not
say that, the payer will deny the claim and the provider faces both
financial loss and potential compliance exposure. The only way to
guarantee trustworthiness is to require every claim to trace back to a
direct quote.*

**Final Decision: Evidence Attribution is non-negotiable. The agent
cannot make any claim without a direct verbatim quote from the source
document, including page number, section, and document name.**

**Human-in-the-Loop**

**Why This Decision:** *Insurance denial decisions are legally and
financially consequential. A 60% confidence threshold is too permissive
for this domain --- a denial recommendation based on uncertain evidence
could trigger costly appeals or patient care delays. The threshold must
be strict enough that any output the agent produces autonomously is
highly reliable.*

**Final Decision: Human review is required for any denial suggestion OR
when confidence score falls below 90%.**

**Phase 2: Architecture Discovery**

**2.1 Agent Framework**

**Why This Decision:** *A single-agent architecture treats verification
as an afterthought --- something checked at the end. But in RCM,
verification needs to be able to send work back for re-extraction if
evidence is missing. This requires a state machine with bidirectional
flow between nodes. LangGraph\'s native support for cyclic graphs and
persistent state makes it the only framework that supports this pattern
cleanly. A hybrid LangChain approach adds complexity without benefit.*

**Final Decision: LangGraph only --- Multi-agent State Machine with
Extractor Node, Auditor Node, Review Loop, and Clarification Node.**

**Multi-Agent State Machine Design**

  ---------------- ---------------------------- -------------------------
  **Node**         **Role**                     **Can Trigger**

  Extractor Node   Reads clinical PDF, extracts Auditor Node
                   relevant evidence with       
                   citations                    

  Auditor Node     Independently validates      Review Loop or Output
                   every citation against       Node
                   source document              

  Review Loop      Sends work back to Extractor Extractor Node (max 3
                   with specific missing        iterations)
                   evidence instructions        

  Clarification    Pauses workflow when text is Resumes Extractor after
  Node             ambiguous, requests human    input received
                   input                        

  Output Node      Formats final verified       End or Human Review queue
                   response with citations and  
                   confidence score             
  ---------------- ---------------------------- -------------------------

**LangGraph State Schema**

The state persists across all nodes and includes:

-   documents_processed: list of pages and sections completed so far

-   extractions: list of evidence items each with direct quote and
    source location

-   audit_results: Auditor Node validation results per citation

-   pending_user_input: boolean flag --- pauses the entire workflow
    without losing any completed work

-   clarification_needed: the specific question for the human when
    ambiguity is detected

-   iteration_count: prevents infinite review loops, capped at 3
    re-extractions

-   confidence_score: final score; triggers human-in-loop queue if below
    90%

*The pending_user_input flag is architecturally critical. Without it,
hitting an ambiguity on page 23 of a 50-page chart discards all previous
work and forces a restart. With it, the agent pauses at exactly that
point, asks the human for clarification, and resumes from page 23 ---
not page 1.*

**2.2 LLM Selection**

**Why This Decision:** *Clinical charts are often 50+ pages. Processing
the entire document in a single context window eliminates chunking
errors and preserves clinical context across sections --- a patient\'s
diagnosis on page 3 must be connected to the procedure on page 47.
Claude 3.5 Sonnet\'s 200K token context window handles this natively.
Additionally, a hybrid cost strategy uses Claude Haiku for cheap
extraction tasks and Claude 3.5 Sonnet only for the high-stakes
verification reasoning --- exactly how production-grade systems stay
profitable.*

**Final Decision: Hybrid model strategy: Claude Haiku for PDF
extraction, Claude 3.5 Sonnet for medical necessity reasoning and final
verification.**

  ------------------ ------------------------ ---------------------------------
  **Task**           **Model**                **Reason**

  PDF text           Claude Haiku             Cheap, fast, sufficient for
  extraction                                  structured extraction

  Policy matching    Claude 3.5 Sonnet        Complex medical reasoning
  and reasoning                               requires full capability

  Final verification Claude 3.5 Sonnet        High-stakes decision --- worth
  and audit                                   the cost

  Embedding          text-embedding-3-small   Cost-effective vector indexing of
  generation                                  policy docs
  ------------------ ------------------------ ---------------------------------

**2.3 Tool Design**

**Why This Decision:** *Insurance payer policies are not available via
clean APIs. They are 200-page PDFs updated quarterly, full of tables,
footnotes, and nested criteria. Simple API calls cannot access this
data. A vector database (Pinecone) with semantic search over chunked
policy documents is the only realistic production approach. Similarly,
clinical PDFs are often scanned, handwritten, or poorly formatted ---
unstructured.io handles messy real-world documents that basic PDF
parsers fail on.*

**Final Decision: Production toolset: unstructured.io for PDF parsing,
Pinecone vector DB for policy search, ICD-10/CPT validator, Microsoft
Presidio for PII scrubbing.**

  ------------------------------- ------------------ -----------------------------------
  **Tool**                        **Library**        **What It Does**

  pdf_extractor(file)             unstructured.io    Extracts text and tables from messy
                                                     scanned clinical PDFs

  policy_search(query)            Pinecone +         Semantic search over indexed payer
                                  embeddings         policy PDFs

  fhir_patient_data(patient_id)   OpenEMR FHIR R4    Retrieves structured diagnoses,
                                                     procedures, encounters

  validate_codes(icd10, cpt)      Custom ICD-10/CPT  Verifies diagnosis codes support
                                  registry           the requested procedure

  citation_verifier(claim, doc)   Custom tool        Confirms claimed quote exists
                                                     verbatim in source document

  pii_scrubber(text)              Microsoft Presidio Scrubs Names/SSNs/DOBs locally
                                                     before text reaches cloud LLM
  ------------------------------- ------------------ -----------------------------------

**2.4 Observability Strategy**

**Why This Decision:** *Generic metrics like latency and token counts do
not tell you if the agent is doing its job in an RCM context. The two
metrics that actually matter are: Faithfulness (is every claim grounded
in a direct source quote?) and Answer Relevancy (did the response
actually address the payer\'s specific coverage criteria?). These
domain-specific metrics directly measure the quality of the output that
determines whether a claim gets approved or denied.*

**Final Decision: LangSmith with domain-specific metrics: Faithfulness,
Answer Relevancy, Citation Accuracy, Review Loop Rate, and Human
Escalation Rate.**

  --------------- ---------------------- ---------------------------------
  **Metric**      **Definition**         **Why It Matters**

  Faithfulness    Every claim grounded   Non-faithful claims cause denials
                  in a direct source     
                  quote                  

  Answer          Response addresses the Irrelevant extractions waste
  Relevancy       payer\'s specific      reviewer time
                  coverage criteria      

  Citation        All cited quotes exist Core quality metric for Evidence
  Accuracy        verbatim in source     Attribution rule
                  document               

  Review Loop     How often Auditor      High rate signals extraction
  Rate            sends work back to     quality issues
                  Extractor              

  Human           How often confidence   Tracks model uncertainty trends
  Escalation Rate falls below 90%        over time

  Latency per     Time in each LangGraph Identifies bottlenecks in the
  Node            node                   multi-agent pipeline
  --------------- ---------------------- ---------------------------------

**Phase 3: Post-Stack Refinement**

**3.1 Failure Mode Analysis**

**Why This Decision:** *Medical data is inherently contradictory.
Different providers document the same patient differently at different
times. An agent that treats contradictions as errors will fail
constantly in real clinical environments. The right approach is to treat
contradictions as data --- flag them explicitly, cite both sources, and
escalate to a human. Similarly, ambiguous laterality (left vs right) is
a common real-world failure that requires a pause-and-ask pattern rather
than a guess.*

**Final Decision: Domain-specific failure handling: contradictions are
flagged as data, ambiguity triggers the Clarification Node, and missing
evidence triggers the Review Loop --- never a guess.**

  --------------- --------------------- ----------------------------------
  **Failure       **Example**           **Agent Response**
  Mode**                                

  Ambiguous       \'Left leg pain\' but Clarification Node →
  clinical text   MRI for \'Right leg\' pending_user_input: true → pause
                                        without losing work

  Contradictory   Monday: \'Stable\',   Flag contradiction → cite both
  notes           Wednesday:            quotes → escalate to human
                  \'Worsening\'         reviewer

  Missing         Payer requires        Auditor triggers Review Loop →
  evidence        \'failed conservative re-extract → if still missing,
                  therapy\' --- not     flag as \'Insufficient
                  documented            Documentation\'

  Unreadable PDF  Scanned handwriting   Mark section \'Extraction Failed\'
  section         or low-resolution     → flag for manual review →
                  scan                  continue with readable sections

  Policy not in   New payer policy not  Return \'Policy Not Found\' →
  vector DB       yet indexed           suggest manual lookup → never
                                        guess

  Max review      Auditor sends back 3  Return partial results with gaps
  loops reached   times with no         explicitly flagged → full chart
                  resolution            escalated to human
  --------------- --------------------- ----------------------------------

**3.2 Security Considerations**

**Why This Decision:** *Patient PII must never reach a cloud LLM.
Microsoft Presidio runs locally, scrubbing Names, SSNs, DOBs, and MRNs
from the text before it is sent to the Anthropic API. This means even if
the cloud LLM were compromised, no patient-identifying data would be
exposed. This is the gold standard approach for HIPAA 2026 compliance
and is architecturally enforced --- the pii_scrubber tool is called
before any other tool in the pipeline.*

**Final Decision: Microsoft Presidio for local PII scrubbing before any
cloud LLM call. HIPAA 2026 compliant architecture.**

-   PII scrubbing runs locally via Microsoft Presidio --- Names, SSNs,
    DOBs, MRNs removed before Claude API call

-   LangSmith traces contain patient IDs only --- never names or
    identifying information

-   Full audit trail: every extraction, citation, and decision logged
    with timestamp and session ID

-   API keys stored in .env, never hardcoded, .gitignore enforced from
    day one

-   OAuth2 tokens stored in memory only --- never written to disk

**3.3 Testing Strategy**

**Why This Decision:** *Generic adversarial tests (prompt injection,
bypass attempts) are necessary but not sufficient for a healthcare
agent. The most dangerous failures in RCM are domain-specific:
conflicting clinical notes, ambiguous anatomical references, and missing
documentation. These require purpose-built test cases that reflect real
clinical scenarios, not just generic AI safety tests.*

**Final Decision: Domain-specific adversarial testing including
contradictory notes, ambiguous laterality, and missing evidence
scenarios --- alongside standard adversarial inputs.**

  ---------------- ----------- ---------------------------------------------
  **Test           **Count**   **Example**
  Category**                   

  Happy path       20+         Complete chart with clear evidence --- expect
                               full citation and approval recommendation

  Missing evidence 10+         Chart lacks \'failed conservative therapy\'
                               --- expect \'Insufficient Documentation\'
                               flag

  Contradictory    10+         \'Stable\' Day 1, \'Worsening\' Day 3 ---
  notes                        expect contradiction flag with both quotes
                               cited

  Ambiguous        5+          \'Left leg pain\' with \'Right knee MRI\' ---
  laterality                   expect Clarification Required status

  Adversarial      10+         Prompt injection, fabrication requests ---
  inputs                       expect refusal and safe fallback

  Low-quality PDFs 5+          Scanned handwritten notes --- expect graceful
                               partial extraction with gaps flagged
  ---------------- ----------- ---------------------------------------------

**3.4 Open Source Contribution**

**Primary: Healthcare RCM Eval Dataset**

-   Release 50+ test cases as a public dataset on HuggingFace or GitHub

-   Each test case includes: input document, expected extractions,
    expected citations, pass/fail criteria

-   Covers: evidence extraction, citation verification, contradiction
    detection, adversarial inputs

-   License: MIT --- free for the healthcare AI community

-   Target: Published by Sunday submission deadline

**Stretch Goal: OpenEMR RCM Agent Package (PyPI)**

-   Package as installable Python library: pip install openemr-rcm-agent

-   Only attempted if all primary deliverables are completed before
    Sunday

**3.5 Deployment & Operations**

  --------------- ---------------- ---------------------------------------
  **Component**   **Technology**   **Rationale**

  Backend API     FastAPI +        Async, fast, auto-generates API docs
                  Uvicorn          

  Agent Framework LangGraph        Multi-agent state machine with review
                                   loops and state persistence

  PII Protection  Microsoft        Local scrubbing before cloud LLM ---
                  Presidio         HIPAA compliant

  Vector DB       Pinecone         Scalable semantic search over payer
                                   policy PDFs

  Observability   LangSmith        Native LangGraph integration,
                                   domain-specific metrics

  Deployment      Railway or       Simple Docker deployment, free tier,
                  Render           public URL for demo

  CI/CD           GitHub Actions   Auto-deploy on push to main, eval suite
                                   runs on every PR
  --------------- ---------------- ---------------------------------------

**Performance Benchmarking Methodology**

Meeting performance targets is not enough --- the methodology for
measuring, validating, and maintaining those targets must be defined
upfront. This section defines how each performance metric is
benchmarked, under what conditions, and what triggers a rollback.

**Benchmarking Layers**

Performance is measured at three distinct layers to isolate bottlenecks:

  ---------------- ------------------------ ------------------------------
  **Layer**        **What Is Measured**     **Tool Used**

  LLM Latency      Time Claude takes to     LangSmith per-node timing
                   respond per node         
                   (Extractor, Auditor)     

  Tool Latency     Time each tool takes:    LangSmith tool trace timing
                   pdf_extractor,           
                   policy_search,           
                   fhir_patient_data        

  End-to-End       Total time from document FastAPI request timing
  Latency          input to verified        middleware
                   response output          

  Network Latency  API call overhead:       httpx instrumentation
                   Pinecone, OpenEMR FHIR,  
                   Claude API               

  Queue Latency    Time document waits      Custom queue timestamp logging
                   before processing begins 
                   in batch mode            
  ---------------- ------------------------ ------------------------------

**Benchmarking Process**

**Step 1: Baseline Measurement**

-   Run 20 representative test documents through the full pipeline
    individually

-   Record latency breakdown per layer for each document

-   Establish baseline: P50, P90, P95, P99 latency for each layer

-   Baseline must be established before any optimization work begins

**Step 2: Load Testing**

-   Simulate concurrent batch processing: 5, 10, 20 simultaneous
    documents

-   Measure latency degradation under load --- target: \<30s maintained
    at 20 concurrent

-   Identify at what concurrency level latency exceeds the 30s target

-   Tool used: Locust for HTTP load testing against FastAPI endpoints

**Step 3: Document Complexity Testing**

-   Test with documents of varying sizes: 10 pages, 25 pages, 50 pages,
    100 pages

-   Measure how latency scales with document length

-   Identify the maximum document size the agent can handle within 30s

-   Documents exceeding limit are automatically split and processed in
    parallel chunks

**Step 4: Regression Testing**

-   Eval suite runs automatically on every GitHub pull request via
    GitHub Actions

-   If P90 latency increases by more than 20% vs baseline, PR is flagged
    for review

-   If eval pass rate drops below 80%, deployment is blocked
    automatically

-   Weekly full benchmark run on production to detect drift over time

**Performance Alerts & Rollback Triggers**

  ------------------ ---------------- -----------------------------------
  **Condition**      **Threshold**    **Action**

  End-to-end latency \> 30s on P90    Alert on-call + investigate
                                      bottleneck layer

  Tool success rate  \< 95%           Alert + switch affected tool to
                                      fallback

  Eval pass rate     \< 80%           Block deployment + rollback to
                                      previous version

  LLM error rate     \> 2%            Alert + activate Claude API
                                      fallback (GPT-4o)

  Latency regression \> 20% increase  Flag PR, require manual approval
                     vs baseline      before merge

  Hallucination rate \> 5%            Immediate rollback + root cause
                                      analysis required
  ------------------ ---------------- -----------------------------------

**Benchmarking Infrastructure**

-   All benchmark results stored in SQLite locally, Postgres in
    production

-   LangSmith dashboard shows real-time latency breakdown per node and
    per tool

-   Weekly benchmark report auto-generated and stored in GitHub
    repository

-   Benchmark test suite is separate from eval suite --- runs on
    realistic document sizes

-   Performance results included in final submission as evidence of
    meeting project targets

**Disaster Recovery & Data Backup Strategy**

Production healthcare systems must plan for failure. This section
defines what happens when components fail, how data is protected, and
how quickly the system can recover. Two key metrics govern this section:

  ---------------------- ---------------------------- -------------------
  **Metric**             **Definition**               **Target**

  RTO (Recovery Time     Maximum acceptable time to   \< 15 minutes for
  Objective)             restore service after a      critical failures
                         failure                      

  RPO (Recovery Point    Maximum acceptable data loss \< 1 hour of
  Objective)             window in case of failure    processing work
                                                      lost
  ---------------------- ---------------------------- -------------------

**Component Failure Scenarios**

  ---------------- ------------------ --------------------------- ----------
  **Component**    **Failure          **Recovery Strategy**       **RTO**
                   Scenario**                                     

  Claude API       Anthropic API      Automatic fallback to       \< 30
                   unreachable or     GPT-4o via LangChain model  seconds
                   rate limited       swap --- no code change     
                                      required                    

  Pinecone Vector  Pinecone service   Fallback to local FAISS     \< 2
  DB               outage             index (snapshot updated     minutes
                                      nightly) --- reduced        
                                      performance but functional  

  OpenEMR FHIR API OpenEMR Docker     Return cached patient data  \< 1
                   instance down      if available; flag as       minute
                                      \'Live Data Unavailable\'   
                                      for human review            

  LangGraph        Mid-document       LangGraph state persisted   \< 5
  workflow crash   processing failure to SQLite at each node ---  minutes
                                      resume from last completed  
                                      node on restart             

  FastAPI server   Application        Railway/Render              \< 2
  crash            process dies       auto-restarts container;    minutes
                                      health check endpoint       
                                      monitored every 30 seconds  

  Full             Hosting provider   Manual failover to backup   \< 15
  infrastructure   down               Railway project in          minutes
  outage                              different region; DNS       
                                      update required             
  ---------------- ------------------ --------------------------- ----------

**Data Backup Strategy**

**LangGraph State Persistence**

-   Agent state is checkpointed to SQLite after every node completion

-   If a workflow crashes on page 23 of 50, it resumes from page 23 ---
    not page 1

-   State checkpoints retained for 30 days for audit and debugging
    purposes

-   Checkpoints are excluded from cloud backups as they contain scrubbed
    (PII-free) data only

**Vector Database Backup**

-   Pinecone index snapshotted nightly to cloud storage (S3 or
    equivalent)

-   Local FAISS index rebuilt from snapshot on startup --- serves as hot
    standby

-   Policy document source PDFs stored in version-controlled cloud
    storage

-   Re-indexing job runs automatically when new payer policies are
    uploaded

**Eval and Benchmark Results**

-   All eval results stored in Postgres with timestamps and version tags

-   LangSmith traces retained for 90 days for debugging and regression
    analysis

-   Weekly database backup exported to GitHub repository as JSON
    artifact

-   Eval dataset (open source contribution) hosted on HuggingFace ---
    inherently backed up

**Audit Logs**

-   Every query, tool call, and response logged with timestamp and
    session ID

-   Audit logs are append-only --- cannot be modified or deleted

-   Retained for minimum 7 years per healthcare compliance requirements

-   Stored separately from application database to prevent accidental
    deletion

**Recovery Runbook**

A documented runbook is maintained in the GitHub repository covering:

-   Step-by-step recovery procedure for each failure scenario above

-   Contact information and escalation path for each failure type

-   Rollback procedure: how to revert to the previous agent version in
    under 5 minutes

-   Post-incident review template: root cause, impact, prevention steps

-   Runbook is tested quarterly with simulated failure drills

**Architecture Summary**

  --------------- --------------------- ----------------------------------
  **Layer**       **Technology**        **Purpose**

  LLM ---         Claude Haiku          Cheap, fast extraction from
  Extraction                            clinical PDFs

  LLM ---         Claude 3.5 Sonnet     Complex medical necessity
  Reasoning                             reasoning and final verification

  Orchestration   LangGraph Multi-Agent Extractor → Auditor → Review Loop
                                        → Output

  State           LangGraph State       pending_user_input flag preserves
  Management      Schema                work on ambiguity

  PDF Processing  unstructured.io       Handles messy scanned clinical
                                        documents

  Policy Search   Pinecone Vector DB    Semantic search over 200-page
                                        payer policy PDFs

  PII Protection  Microsoft Presidio    Local scrubbing before any cloud
                                        LLM call

  Code Validation Custom ICD-10/CPT     Validates diagnosis-procedure code
                  tool                  alignment

  Observability   LangSmith             Faithfulness, relevancy, citation
                                        accuracy metrics

  API Server      FastAPI + Uvicorn     HTTP endpoints for agent
                                        interaction

  Open Source     Eval Dataset on       50+ RCM test cases released
                  HuggingFace           publicly
  --------------- --------------------- ----------------------------------

*This Pre-Search document was completed before writing any production
code, following the AgentForge Pre-Search methodology. Every
architectural decision is documented with its rationale so that the
reasoning is transparent and defensible in review.*
