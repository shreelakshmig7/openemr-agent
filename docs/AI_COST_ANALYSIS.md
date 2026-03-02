# AI Cost Analysis — AgentForge Healthcare RCM Agent

**API Key:** `shree-collab-success`  
**Period covered:** Feb 23 – Mar 1, 2026 (7-day development sprint)  
**Author:** Shreelakshmi Gopinatha Rao  
**Deployed:** [Railway](https://openemr-agent-production.up.railway.app)

---

## 1. Development & Testing Costs (Actual)

### 1.1 Raw spend by day and model

All figures derived from the Anthropic usage CSV export (`claude_api_tokens_2026_02_23.csv`).  
The sprint began on Sonnet 4 (`claude-sonnet-4-20250514`) and upgraded to Sonnet 4.5 (`claude-sonnet-4-5-20250929`) on Feb 27 — the day the multi-agent LangGraph graph was fully assembled and the first end-to-end eval suite run was executed.

| Date | Model | Input tokens | Output tokens | Input cost | Output cost | Day total |
|------|-------|-------------|--------------|-----------|------------|-----------|
| Feb 23 | Claude Haiku 4.5 | 69,483 | 1,462 | $0.069 | $0.007 | $0.077 |
| Feb 23 | Claude Sonnet 4 | 410,839 | 13,519 | $1.233 | $0.203 | $1.436 |
| **Feb 23 total** | | | | | | **$1.513** |
| Feb 24 | Claude Sonnet 4 | 402,278 | 47,545 | $1.207 | $0.713 | **$1.920** |
| Feb 25 | Claude Sonnet 4 | 465,878 | 43,393 | $1.398 | $0.651 | **$2.049** |
| Feb 26 | Claude Haiku 4.5 | 25,865 | 2,091 | $0.026 | $0.010 | $0.036 |
| Feb 26 | Claude Sonnet 4 | 173,888 | 14,788 | $0.522 | $0.222 | $0.744 |
| **Feb 26 total** | | | | | | **$0.780** |
| Feb 27 | Claude Haiku 4.5 | 825,149 | 62,794 | $0.825 | $0.314 | $1.139 |
| Feb 27 | Claude Sonnet 4.5 | 617,445 | 87,987 | $1.852 | $1.320 | $3.172 |
| **Feb 27 total** | | | | | | **$4.311** |
| Feb 28 | Claude Haiku 4.5 | 29,988 | 2,078 | $0.030 | $0.010 | $0.040 |
| Feb 28 | Claude Sonnet 4.5 | 29,737 | 4,223 | $0.089 | $0.063 | $0.153 |
| **Feb 28 total** | | | | | | **$0.193** |
| Mar 1 | Claude Haiku 4.5 | 1,214,820 | 45,176 | $1.215 | $0.226 | $1.441 |
| Mar 1 | Claude Sonnet 4.5 | 476,579 | 61,765 | $1.430 | $0.926 | $2.356 |
| **Mar 1 total** | | | | | | **$3.797** |

**Grand total (Anthropic API, 7-day sprint): $14.56**

---

### 1.2 Model totals (cumulative, all sessions)

| Model | Dates active | Input tokens | Output tokens | Input cost | Output cost | **Subtotal** |
|-------|-------------|-------------|--------------|-----------|------------|-------------|
| Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Feb 23, 26, 27, 28, Mar 1 | 2,165,305 | 113,601 | $2.165 | $0.568 | **$2.733** |
| Claude Sonnet 4 (`claude-sonnet-4-20250514`) | Feb 23–26 | 1,452,883 | 119,245 | $4.359 | $1.789 | **$6.148** |
| Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) | Feb 27–Mar 1 | 1,123,761 | 153,975 | $3.371 | $2.310 | **$5.681** |
| **Grand total** | | **4,741,949** | **386,821** | **$9.895** | **$4.667** | **$14.562** |

*Minor rounding differences between per-day and aggregate figures are due to token-level precision.*

Sonnet (combined Sonnet 4 + Sonnet 4.5) accounts for **$11.83 (81.2%)** of total API spend. Haiku's 2.17M input tokens at $1.00/MTok cost only **$2.73 (18.8%)** — the routing/classification layer is cheap relative to the synthesis layer.

Pricing used: **Haiku 4.5** at $1.00/MTok input · $5.00/MTok output; **Sonnet 4** at $3.00/MTok input · $15.00/MTok output; **Sonnet 4.5** at $3.00/MTok input · $15.00/MTok output.

---

### 1.3 Token consumption

**Cumulative across all sessions: ~5.13 million tokens**

| Model | Input tokens | Output tokens | Total tokens |
|-------|-------------|--------------|-------------|
| Claude Haiku 4.5 | 2,165,305 | 113,601 | 2,278,906 |
| Claude Sonnet 4 | 1,452,883 | 119,245 | 1,572,128 |
| Claude Sonnet 4.5 | 1,123,761 | 153,975 | 1,277,736 |
| **Total** | **4,741,949** | **386,821** | **5,128,770** |

The output-to-input ratio is low (1:12 overall) because AgentForge queries involve large system prompts, HIPAA safety rules, EHR data payloads, and extracted PDF content passed as input context — while outputs are structured clinical responses and JSON tool plans.

---

### 1.4 Estimated API call volume

Each AgentForge query routes through up to 5 nodes:  
**Router (Haiku)** → **Orchestrator (Haiku)** → **Extractor (tools only)** → **Auditor (Sonnet)** → **Output (Sonnet)**

Some Extractor queries also trigger a small Haiku call for drug name extraction (safety check queries). Average Haiku tokens per query: ~1,700 input, ~150 output. Average Sonnet tokens per query: ~4,500 input, ~750 output (higher on PDF and multi-turn auditor iterations).

| Model | Total input tokens | Avg tokens/call | Est. API calls |
|-------|--------------------|----------------|---------------|
| Haiku 4.5 | 2,165,305 | ~1,700 | **~1,274** |
| Sonnet 4 + 4.5 (combined) | 2,576,644 | ~4,500 | **~572** |
| **Total** | | | **~1,846** |

The Haiku:Sonnet call ratio is ~2.2:1, matching the architecture exactly — each query generates 2 Haiku calls (Router + Orchestrator) and 1–2 Sonnet calls (Auditor + Output). Safety check queries with drug-name extraction add a third lightweight Haiku call (max 16 output tokens, capped in code).

The large Haiku token spikes on Feb 27 (825K) and Mar 1 (1.21M) indicate **~8 full eval-suite runs on Feb 27 and ~11 runs on Mar 1** — consistent with iterating on failing test cases in the 63-case golden dataset and re-running the suite to confirm fixes.

---

### 1.5 Daily spend pattern and what it reflects

| Date | Total spend | Development phase |
|------|------------|------------------|
| Feb 23 | $1.51 | Initial agent scaffolding — Router, Orchestrator, and Extractor nodes wired up under Sonnet 4. First live FHIR queries against OpenEMR. Haiku appears for routing classification only. |
| Feb 24 | $1.92 | Auditor + Output nodes built. High Sonnet output (47K tokens) reflects iterative prompt engineering on clinical synthesis and citation verification. |
| Feb 25 | $2.05 | **Highest Sonnet 4 day** (465K input). PDF eval suite development — PriorAuth and ClinicalNote test PDFs integrated, 3-gate source integrity stack implemented, unstructured.io PDF extraction wired. |
| Feb 26 | $0.78 | Integration polish, HITL sync-confirmation flow, PII scrubbing (Presidio). Lower spend as architecture settled. |
| Feb 27 | $4.31 | **Model upgrade to Sonnet 4.5 + first major eval run.** 825K Haiku tokens = ~8 full 63-case eval-suite runs. Sonnet 4.5's higher output token count (87K — highest single-model daily output in sprint) reflects richer clinical synthesis on harder cases. |
| Feb 28 | $0.19 | Minimal targeted debugging. Light session fixing specific edge cases from Feb 27 eval results. |
| Mar 1 | $3.80 | **Final comprehensive eval run.** 1.21M Haiku tokens = ~11 full eval-suite runs — largest Haiku day of the sprint, confirming iterative regression-testing before submission. Sonnet 4.5 output (61K) reflects varied case complexity across all 63 tests. |

The 48% drop from the highest day (Feb 27: $4.31) to the sprint average ($2.08) reflects the cost of bulk eval-suite validation being front-loaded at model upgrade and sprint close — rather than a structural inefficiency.

---

### 1.6 Supporting service costs (actual)

| Service | Usage during sprint | Cost | Notes |
|---------|-------------------|------|-------|
| **Unstructured.io** | 28 PDF eval cases × ~3 pages × ~19 eval runs ≈ ~1,596 pages processed | **$0.00** | Free tier: 15,000 pages included. Sprint used ~10.6% of allowance. $0.03/page applies beyond 15K. |
| **Voyage AI** (`voyage-large-2`) | 11 policy-search eval cases × ~19 eval runs ≈ ~209 embedding calls; ~20 tokens/query | **~$0.001** | Negligible. $0.12/MTok; 209 × 20 tokens = 4,180 tokens = $0.0005. |
| **Pinecone** (Serverless) | ~209 vector queries; `agentforge-rcm-policies` index, us-east-1 | **$0.00** | Starter free tier. Serverless reads at $8.25/M RU — dev scale consumed <1K RUs total. |
| **Railway** | AgentForge FastAPI server deployed throughout sprint | **~$0–5/mo** | Starter plan — usage metrics not exposed on this tier. Cost is minimal fixed overhead. |
| **LangSmith** | Full observability traces for all 1,846+ API calls | **$0.00** | Developer plan free tier. Trace storage and replay used throughout sprint. |

---

### 1.7 Other AI-related development costs

| Item | Cost | Notes |
|------|------|-------|
| Cursor Pro (IDE + Claude integration) | $20/month | Primary development environment. Claude Sonnet used via Cursor agent/plan mode for multi-file code generation, refactoring, and architecture consultation. |
| Claude.ai | $0 | Trial plan — used for prompt design review and agent behavior inspection outside the IDE. |
| Unstructured.io | $0 | Free tier (15,000 pages). Sprint consumed ~1,600 pages. |
| Pinecone | $0 | Starter free tier sufficient for development index. |
| Voyage AI | ~$0 | Sub-cent spend; well within any free credits. |
| Railway | ~$0–5/month | Starter plan hosting for deployed agent. |
| **Total dev tooling** | **~$20–25/month** | Fixed overhead, independent of API usage. |

**Total development cost for the sprint: $14.56 (Anthropic API) + $0 (Unstructured/Pinecone/Voyage — free tiers) + ~$20 (Cursor Pro) ≈ $34.56–39.56**

---

## 2. Production Cost Projections

### 2.1 Pricing reference

| Service | Unit | Price |
|---------|------|-------|
| Claude Haiku 4.5 | Input | $1.00 / MTok |
| Claude Haiku 4.5 | Output | $5.00 / MTok |
| Claude Sonnet 4.5 | Input | $3.00 / MTok |
| Claude Sonnet 4.5 | Output | $15.00 / MTok |
| Unstructured.io | Per page (PDF extraction) | $0.03 / page |
| Voyage AI (`voyage-large-2`) | Embedding tokens | $0.12 / MTok |
| Pinecone (Serverless) | Read units | $8.25 / M RU |

*Prompt caching (Anthropic beta) not yet enabled in production — would reduce input costs 90% on cached blocks. See Section 2.7.*

---

### 2.2 Per-query token model

Each AgentForge query routes: **Router (Haiku) → Orchestrator (Haiku) → Extractor (tools) → Auditor (Sonnet) → Output (Sonnet)**. Token counts below reflect the system prompt, clinical safety rules, EHR payloads, and conversation history that accumulate as input context.

#### Query Type 1 — Standard clinical query (medication list, allergy lookup)

Single patient lookup + EHR FHIR data, no PDF, no policy.

| Component | Model | Tokens |
|-----------|-------|--------|
| Router system prompt + 6-intent schema + query | Haiku | ~800 input, ~50 output |
| Orchestrator system + intent fields + query | Haiku | ~900 input, ~100 output |
| Extractor | Tools only (FHIR) | — |
| Auditor: extractions + citation task | Sonnet | ~2,000 input, ~300 output |
| Output: CLINICAL_SAFETY_RULES + extractions + query | Sonnet | ~2,200 input, ~400 output |
| **Totals** | | **3,900 Haiku input / 150 Haiku output · 4,200 Sonnet input / 700 Sonnet output** |

**Cost:** 3,900×$0.000001 + 150×$0.000005 + 4,200×$0.000003 + 700×$0.000015  
= $0.0039 + $0.00075 + $0.0126 + $0.0105 = **~$0.027 per query**

---

#### Query Type 2 — Drug interaction / safety check (multi-tool EHR)

Patient + medications + interactions + allergy cross-reference. May include a third Haiku call for drug-name extraction (max 16 output tokens).

| Component | Model | Tokens |
|-----------|-------|--------|
| Router + Orchestrator | Haiku | ~1,700 input, ~160 output |
| Drug-name extraction (safety check only) | Haiku | ~400 input, ~16 output |
| Auditor: interaction + allergy extractions | Sonnet | ~3,500 input, ~400 output |
| Output: synthesis + severity verdict | Sonnet | ~3,000 input, ~500 output |
| **Totals** | | **2,100 Haiku input / 176 Haiku output · 6,500 Sonnet input / 900 Sonnet output** |

**Cost:** 2,100×$0.000001 + 176×$0.000005 + 6,500×$0.000003 + 900×$0.000015  
= $0.0021 + $0.00088 + $0.0195 + $0.0135 = **~$0.036 per query**

---

#### Query Type 3 — Prior authorization / payer policy search

Patient EHR + Voyage AI embedding + Pinecone policy retrieval + Sonnet synthesis.

| Component | Model / Service | Tokens / Units |
|-----------|----------------|----------------|
| Router + Orchestrator | Haiku | ~1,700 input, ~150 output |
| Extractor: `policy_search` tool → Voyage embed → Pinecone query | Voyage AI | ~100 embedding tokens |
| Auditor: EHR + policy content | Sonnet | ~4,000 input, ~400 output |
| Output: prior-auth criteria + verdict | Sonnet | ~3,500 input, ~500 output |

**Cost:** (1,700×$0.000001 + 150×$0.000005) + (100×$0.00000012) + (7,500×$0.000003 + 900×$0.000015)  
= $0.00245 + $0.000012 + $0.0225 + $0.0135 = **~$0.038 per query**

---

#### Query Type 4 — PDF clinical note review (Unstructured.io + EHR)

Clinical PDF attachment processed by unstructured.io, then cross-referenced against EHR data.

| Component | Model / Service | Tokens / Units |
|-----------|----------------|----------------|
| Router + Orchestrator | Haiku | ~1,700 input, ~150 output |
| PDF extraction | Unstructured.io | ~3 pages → $0.09 |
| Extractor: `pdf_extractor` + EHR tools | Tools only | — |
| Auditor: large PDF context + extractions | Sonnet | ~8,000 input, ~600 output |
| Output + 3-gate source integrity | Sonnet | ~5,000 input, ~600 output |

**Cost:** $0.00245 + $0.09 + (13,000×$0.000003 + 1,200×$0.000015)  
= $0.00245 + $0.09 + $0.039 + $0.018 = **~$0.149 per query**

*Unstructured.io PDF extraction ($0.09) is the dominant cost for this query type — 60% of the total.*

---

#### Query Type 5 — Full HITL prior-auth review (PDF + EHR + policy + HITL confirmation)

The most complex path: PDF ingestion, EHR data, policy retrieval, Auditor iteration loop (up to 3×), and human-in-the-loop sync confirmation before writing back to OpenEMR FHIR.

| Component | Model / Service | Tokens / Units |
|-----------|----------------|----------------|
| Router + Orchestrator | Haiku | ~2,000 input, ~200 output |
| PDF extraction | Unstructured.io | ~3 pages → $0.09 |
| Auditor (up to 3 iterations with citation retry) | Sonnet | ~18,000 input, ~1,500 output |
| Output + Comparison nodes | Sonnet | ~5,500 input, ~800 output |

**Cost:** $0.003 + $0.09 + (23,500×$0.000003 + 2,300×$0.000015)  
= $0.003 + $0.09 + $0.0705 + $0.0345 = **~$0.198 per query**

---

### 2.3 Usage assumptions

Scale tiers sourced directly from the AgentForge Pre-Search document (Section 1.2), which defined deployment targets before any code was written. The unit is **charts/month** — one chart = one agent invocation (a full clinical document review or structured clinical query).

| Assumption | Value | Source |
|------------|-------|--------|
| Charts per user per month | 50 | Pre-Search Section 1.2 — batch throughput target of 20–50 charts/hour defines realistic daily usage |
| Latency target per chart | < 30 seconds | Pre-Search Section 1.2 — accuracy over speed; RCM denials cost $25–200 to rework |
| Chart mix — standard clinical (EHR) | 40% | Medication lists, allergy checks, patient demographics — most frequent daily lookup |
| Chart mix — drug interaction / safety check | 25% | Before-you-prescribe safety verification; triggered by clinical workflow |
| Chart mix — prior auth / policy search | 20% | Payer coverage lookup; primary use case for RCM billing staff |
| Chart mix — PDF clinical note review | 10% | Prior auth letters, discharge summaries uploaded for evidence extraction |
| Chart mix — full HITL prior-auth review | 5% | Complex cases requiring human confirmation before FHIR write-back |
| PDF pages per chart (PDF-type) | ~3 pages avg | Based on AgentForge_Test_PriorAuth.pdf and AgentForge_Test_ClinicalNote.pdf test documents |

---

### 2.4 Weighted average cost per chart

```
= 0.40 × $0.027   (standard clinical / Haiku + Sonnet)
+ 0.25 × $0.036   (drug interaction / Haiku + Sonnet)
+ 0.20 × $0.038   (prior auth / + Voyage + Pinecone)
+ 0.10 × $0.149   (PDF note review / + Unstructured.io)
+ 0.05 × $0.198   (full HITL prior-auth)

= $0.0108 + $0.009 + $0.0076 + $0.0149 + $0.0099

= $0.052 per chart (weighted average)
```

**Monthly AI cost per active clinical user: 50 charts × $0.052 = $2.60/user/month**

PDF processing (Unstructured.io at $0.09 per 3-page chart) is the single largest per-chart cost driver for PDF-involved cases — contributing $0.024 to the weighted average despite only 15% chart share. Without PDF charts, the weighted average drops to **$0.032/chart ($1.60/user/month)**.

---

### 2.5 Monthly cost projections

Scale tiers match the Pre-Search document exactly. Unstructured.io free tier (15,000 pages) covers pilot and early small-clinic scale; paid pages begin when 15% PDF mix × charts × 3 pages exceeds 15,000 (at ~33,000 charts/month). Pinecone Standard ($70/mo) required above small-clinic scale for production SLA.

| Scale | Users | Charts/mo | Anthropic API | Unstructured.io | Pinecone | Railway | **Actual total** | **Pre-Search estimate** |
|-------|-------|----------|--------------|----------------|---------|---------|-----------------|------------------------|
| **Pilot** | 100 | 5,000 | $260 | $0 (2,250 pages — within free tier) | $0 (free) | $5 | **~$265/month** | $80–150/month |
| **Small clinic** | 1,000 | 50,000 | $2,600 | $225 (7,500 paid pages above free tier) | $70 | $20 | **~$2,915/month** | $600–1,200/month |
| **Mid-size hospital** | 10,000 | 500,000 | $26,000 | $6,300 (210,000 paid pages) | $70 | $50 | **~$32,420/month** | $5,000–9,000/month |
| **Enterprise** | 100,000 | 5,000,000 | $260,000 | $67,500 (2.25M pages) | $500 | $500 | **~$328,500/month** | $40,000–70,000/month |

**Why actual costs exceed Pre-Search estimates:** The Pre-Search was written before detailed pricing research. Three factors drove the gap: (1) Haiku 4.5 released at $1.00/MTok — higher than the $0.25/MTok assumed for Haiku 3.5; (2) Sonnet 4.5 launched at $3.00/MTok input vs ~$1.50/MTok assumed; (3) Unstructured.io at $0.03/page was not fully accounted for in the pre-search cost model. The per-chart cost of $0.052 vs the pre-search implied $0.016–0.030/chart represents a 2–3× variance — expected at this stage of estimation.

*Unstructured.io free tier (15,000 pages) covers the full pilot tier (5,000 charts × 15% PDF × 3 pages = 2,250 pages). Paid tier begins mid-way through the small-clinic scale.*

---

### 2.6 Per-user monthly cost summary

| Scale | Charts/user/mo | API + PDF cost/user | Infra cost/user | **Total cost/user** | Pre-Search est/user |
|-------|---------------|--------------------|-----------------|--------------------|---------------------|
| Pilot (100 users) | 50 | $2.60 | $0.05 | **$2.65** | $0.80–1.50 |
| Small clinic (1,000 users) | 50 | $2.60 | $0.32 | **$2.92** | $0.60–1.20 |
| Mid-size hospital (10,000 users) | 50 | $2.60 | $0.64 | **$3.24** | $0.50–0.90 |
| Enterprise (100,000 users) | 50 | $2.60 | $0.69 | **$3.29** | $0.40–0.70 |

The Anthropic API cost ($2.60/user/month) is fixed and linear at all scales — the dominant cost driver. Infrastructure (Unstructured.io PDF pages + Pinecone + Railway) grows sub-linearly. At a typical healthcare SaaS pricing point of **$30–100/user/month for a clinical AI assistant**, this yields a **9×–30× margin over total AI cost** — commercially viable at every tier. The pre-authorization denial rework cost alone ($25–200 per denied claim, per Pre-Search Section 1.3) means a single avoided denial pays for 10–77 months of per-user AI cost.

---

### 2.7 Cost optimization levers

| Strategy | Estimated saving | Implementation effort |
|----------|-----------------|----------------------|
| **Anthropic Prompt Caching** for system prompt + CLINICAL_SAFETY_RULES (~600 static tokens shared across every call) | ~20–30% reduction on input tokens (cache-read at $0.10/MTok vs $1.00/MTok Haiku, $0.30/MTok vs $3.00/MTok Sonnet) | Medium — add `cache_control` headers to static message blocks in Orchestrator + Output system prompts |
| **Session-level EHR caching** (Layer 2 cache already built into state) | 40–60% reduction on FHIR tool calls for follow-up queries in same session; Extractor skips redundant tools | Already partially implemented — `extracted_patient` and `payer_policy_cache` in AgentState prevent re-fetching within a session |
| **Shift more out-of-scope queries to Router short-circuit** (expand `OUT_OF_SCOPE` intent set) | Proportional to non-clinical query volume — OUT_OF_SCOPE queries skip Orchestrator + Extractor + Auditor entirely; cost ≈ $0.001/query (Router only) | Low — tune Router system prompt; already implemented for non-healthcare queries |
| **Unstructured.io page caching** — cache PDF extraction result by file hash (already implemented as `extracted_pdf_hash` in AgentState for session) | Eliminates repeat Unstructured.io charges when same PDF is re-referenced in conversation | Already implemented within session scope — extend to persistent SQLite cache across sessions for the same document |
| **Batch API (Anthropic)** for eval runs and async RCM workflows | 50% cost reduction on all eval suite runs; applicable to non-real-time batch prior-auth processing | Low for evals — use `anthropic.Batch` API for the eval runner; higher effort for production async path |
| **Downgrade Auditor to Haiku for low-confidence retrieval miss** (when extractions are empty and confidence < 0.5) | ~$0.020 saved per empty-result query (skip Sonnet Auditor synthesis) | Low — add confidence gate in `_route_from_auditor` before Sonnet call |

**With prompt caching + Batch API for evals, realistic development cost drops ~35–40%**: the $14.56 sprint cost becomes ~$8.74–9.46 if those two optimizations were active. Production cost per user drops from $1.56 to ~$1.10–1.25/month.

---

## 3. Key Takeaways

- **Total development cost for the sprint: ~$34.56–39.56** — $14.56 in Anthropic API (7 days, 5.13M tokens across 3 model versions), $0 across Unstructured.io / Pinecone / Voyage AI / LangSmith (all within free tiers), and ~$20 in Cursor Pro. A production-grade healthcare multi-agent system — 8 nodes, 8 tools, HITL, FHIR integration, 63-case eval suite — built for under $40 in AI spend.

- **Haiku does the heavy lifting on volume, Sonnet carries the cost.** Haiku handled 2.17M input tokens (Router + Orchestrator + drug-name extraction) at $2.73 total — 18.8% of spend for ~69% of all API calls. Sonnet handled 2.58M input tokens at $11.83 — 81.2% of spend for the two highest-stakes nodes (Auditor + Output), where clinical reasoning quality is non-negotiable. The hybrid cost strategy from the Pre-Search doc is validated by the data.

- **The Feb 27 model upgrade (Sonnet 4 → Sonnet 4.5) is visible in the data.** Sonnet 4.5 output tokens on Feb 27 (87,987) are 60–80% higher than any Sonnet 4 day, reflecting richer synthesis responses on complex clinical queries — a quality upgrade at the same per-token price ($3.00/MTok input).

- **Eval-suite runs dominate Haiku token volume.** The Feb 27 (825K) and Mar 1 (1.21M) Haiku spikes correspond to ~8 and ~11 full 63-case eval runs respectively. Running the eval suite is cheap: $0.001 per test case in Haiku routing costs. Rigorous automated testing against the golden dataset is the right investment — a single clinical error in production (a missed allergy conflict, a hallucinated policy citation) costs far more than the entire eval budget.

- **PDF processing (Unstructured.io) is the per-chart cost cliff in production.** At $0.09 per 3-page PDF, a PDF chart costs 5.5× more than a standard EHR query ($0.149 vs $0.027). The existing `extracted_pdf_hash` session cache in AgentState already eliminates repeat charges when the same document is referenced within a session — extending this to cross-session SQLite persistence is the highest-ROI single optimization available.

- **Production cost is $2.60/user/month at 50 charts/month** (Pre-Search deployment target) — dominated by the Anthropic API and linear at all scales. Infrastructure (Pinecone, Railway, Unstructured.io) is negligible until mid-size hospital scale. At a $30–100/user/month healthcare SaaS price point, total AI cost yields a **9×–30× margin**. A single avoided prior-authorization denial ($25–200 rework cost per the Pre-Search) pays for 10–77 months of per-user AI spend.

- **Actual costs are 2–4× above Pre-Search estimates.** The Pre-Search projected $80–150/month for a 100-user pilot; actual is ~$265/month. The gap traces to three pricing changes that occurred after the pre-search was written: Haiku 4.5 at $1.00/MTok (vs ~$0.25/MTok for Haiku 3.5 assumed), Sonnet 4.5 at $3.00/MTok, and Unstructured.io at $0.03/page being omitted from early estimates. The variance is expected at pre-code estimation stage and does not affect commercial viability.

- **Enterprise scale (100K users, 5M charts/month) costs ~$328,500/month** — $260K Anthropic + $67.5K Unstructured.io PDF + $1K infra. That figure targets the $265B U.S. prior-authorization denial market directly. With prompt caching (−25%) and Batch API for offline processing (−50% on async workflows), that figure drops to **~$200,000–240,000/month** — and every 1% reduction in denial rate for a mid-size hospital network returns millions in recovered revenue.
