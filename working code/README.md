# Document Comparison LLM Evaluation Pipeline

Evaluates how well an LLM **compares two documents**, using a golden dataset built from the documents
themselves, a parent-child RAG with one isolated vector store per document, and the metrics of
`LLM_Evaluation_Framework.xlsx` (Performance P1-P7, Reliability R1-R4, Document Comparison column).

```
Phase 0  ingest       documents -> sections (parents) + small chunks (children)
Phase 1  align        matching sections + fact pairs (SAME / DIFFERENT / PARTIAL / ONE_SIDED)
Phase 2  golden       golden questions (differential / equivalence / asymmetric) with traceable evidence
Phase 2b reliability  variants of the golden questions (paraphrase, typos, adversarial, long context, ...)
Phase 3  rag          the model under test answers, with chunk-ID citations
Phase 4  evaluate     Performance metrics (P1-P7) + retrieval + comparison checks
Phase 4b reliability  Reliability metrics (R1-R4) vs the clean run
Phase 5  report       Excel scorecard (framework layout) + HTML report
```

---

## 1. What you need

| Item | Notes |
|---|---|
| Python **3.11 or newer** | Tested on Python 3.14 (Windows 11). Check with `python --version`. |
| An **API key with billing enabled** | OpenAI (tested) or Google Gemini. A Gemini *free-tier* key allows only ~20 requests/day/model - far too few. ChatGPT / Google AI subscriptions do **not** include API usage. |
| ~3 GB free disk | PyTorch + the embedding model (all-MiniLM-L6-v2, ~90 MB) + the NLI model (~700 MB), downloaded from Hugging Face on first use. |
| Internet access | API calls, and the one-time model downloads. Offline machine: download both models elsewhere and set `models.embeddings` / `evaluation.nli_model` to their local folder paths. |

---

## 2. Copy the project

Copy these to the new machine (keep the folder structure):

```
config.yaml
requirements.txt
.env.example
.gitignore
README.md
LLM_Evaluation_Framework.xlsx      # the framework workbook (used by Phase 5)
src/                               # all code
scripts/                           # one script per phase
<your two documents>               # .pdf, .docx, .txt or .md
```

Do **not** copy `.env` (it holds your API keys - create a new one, step 4).

Optional: copy `data/` and `outputs/` too if you want to keep previous results. Every phase resumes from
what is already saved, so copied results are reused instead of paid for again. `data/vectorstore/` is
rebuilt automatically if it is missing.

---

## 3. Create a virtual environment and install packages

Open a terminal **in the project folder**.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked ("running scripts is disabled"), run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the install:
```bash
python -c "import pymupdf, chromadb, langchain_openai, sentence_transformers, openpyxl; print('ok')"
```

Check the embedding setup (local all-MiniLM-L6-v2 only - no LLM or API calls; run after Phase 0):
```bash
python scripts/check_embeddings.py
```
It blanks the API keys while running, builds temporary vector stores from the Phase 0 chunks, and
reports retrieval Recall@k on the golden questions if Phase 2 has run. Expect `13/13 checks passed`.

Activate the environment (`.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`) every time you open
a new terminal.

---

## 4. Add your API key

Copy `.env.example` to `.env` and fill in the key for the provider you use:

```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

Never commit or share `.env` (it is already in `.gitignore`).

---

## 5. Configure `config.yaml`

Open `config.yaml` and check these blocks - everything else has working defaults.

**a) Your documents**
```yaml
documents:
  - id: D1                     # short internal code (letters/digits/_), never shown in questions
    label: UK Handbook         # how answers refer to the document
    path: my_first_document.pdf
  - id: D2
    label: India Handbook
    path: my_second_document.docx
alignment:
  doc_a: D1                    # the two documents to compare
  doc_b: D2
```

**b) Provider and models**
```yaml
models:
  provider: openai             # or google
  openai:
    generator: gpt-5.4-mini    # builds the dataset (Phases 1-2)
    under_test: gpt-4.1-mini   # the model being evaluated (Phase 3)
    judge: gpt-5.4-mini        # LLM judge (Phases 2, 4, 4b) - keep it different from under_test
    embeddings: text-embedding-3-large
```
If a model name is rejected ("model not found"), list the models your key can use and update the names:
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from openai import OpenAI; print(sorted(m.id for m in OpenAI().models.list()))"
```

**c) Words that would reveal which document is which** - **must be updated for new documents**:
```yaml
golden:
  banned_terms: [UK, United Kingdom, England, India, Indian, ...]   # countries, cities, entity names of YOUR documents
```
Document ids, labels, file names and generic patterns ("Region 1", "R1", "page 3", "Document A") are
always blocked automatically.

**d) Optional:** pass thresholds (`evaluation.pass_thresholds`, `reliability.pass_thresholds`) and token
prices for the cost metric (`evaluation.pricing_usd_per_1m_tokens` - check current provider prices).

> **Subject area:** the prompts are written for HR handbooks ("HR policy analyst", "employee"). For other
> kinds of documents (contracts, product specs, ...) edit the wording in `src/*/prompts.py`.

### Changing the LLM

All models are set in the `models:` block of `config.yaml` - no code changes for OpenAI or Gemini models.

| To change... | Edit | Then re-run |
|---|---|---|
| **The model being evaluated** (most common) | `under_test` | Phase 3 with **new run ids** -> Phase 4 -> 4b -> 5. Phases 0-2b are reused. |
| The LLM judge | `judge` (keep it different from `under_test`) | Phase 4 -> 4b -> 5 (judge verdicts are cached per judge model). |
| The dataset builder | `generator` (and `golden.cross_topic_model`) | Phase 1 and 2 with `--fresh`, then everything after. |
| Embeddings | `models.embeddings` (one setting for everything; default local `sentence-transformers/all-MiniLM-L6-v2`, 384 dims, no API - a local folder path also works offline) | Phase 3 onwards (vector stores rebuild automatically). Re-calibrate the cosine thresholds in `evaluation` / `reliability` for a new embedding model. |
| Provider | `provider: openai` or `google`, plus the model names under that provider | As above for the roles that changed. |

To compare several models, run Phases 3-5 once per model with different run ids
(e.g. `--run-id baseline_gpt-4.1-mini`, `--run-id baseline_gpt-5.4-mini`) - each gets its own scorecard.
Other providers (Anthropic, Azure OpenAI, local models) need a few lines in `get_chat` / `get_embedder`
in `src/common/llm.py`.

---

## 6. Run the pipeline, phase by phase

Run every command from the project folder with the virtual environment active. Each phase reads the
previous phase's saved output, so run them in order. **Stop at the review checkpoints** - errors caught
early are cheap; errors caught late have already been paid for downstream.

### Phase 0 - ingest (no API calls)
```bash
python scripts/run_phase0_ingest.py
```
✅ Expect: `checks PASSED` for both documents and word coverage ~100%.
🔍 **Review** `data/processed/D1/structure_review.md` and `.../D2/structure_review.md`: are the sections
(parents) the real sections of the document? If not, see Troubleshooting.

### Phase 1 - topic alignment and fact pairs
```bash
python scripts/run_phase1_align.py
```
✅ Expect: a topic list where each section is matched to its counterpart, then `failures: none`.
🔍 **Review** `data/processed/alignment/alignment_review.md`: topic map + every fact pair with each
document's value and the relation. Pairs with flags were auto-corrected or need a look.

### Phase 2 - golden dataset
```bash
python scripts/run_phase2_golden.py
```
✅ Expect: `Accepted N | rejected M` with all three question types present.
🔍 **Review** `data/golden/golden_review.md` (accepted questions + rejection reasons). Remove any weak
question from `data/golden/golden.jsonl` if needed.

### Phase 2b - reliability test suite
```bash
python scripts/run_phase2b_reliability_suite.py
```
✅ Expect: `~450 variants` across R1-R4. Examples in `data/reliability/suite_review.md`.

### Phase 3 - RAG answers by the model under test
First check retrieval quality (no LLM cost):
```bash
python scripts/run_phase3_rag.py --retrieval-only
```
✅ Expect: `retrieval gate passed` (child recall@5 >= 0.8 by default).

Then the two runs (choose your own run ids; re-using an id resumes that run):
```bash
python scripts/run_phase3_rag.py --run-id baseline
python scripts/run_phase3_rag.py --questions data/reliability/variants.jsonl --run-id reliability
```
Tip: `--limit 3 --run-id smoke` runs 3 questions as a quick test.

### Phase 4 - Performance evaluation (clean run)
```bash
python scripts/run_phase4_evaluate.py --run-id baseline
```
The first run downloads the NLI model (~700 MB). ✅ Expect a P1-P7 table with PASS/FAIL.

### Phase 4b - Reliability evaluation
```bash
python scripts/run_phase4b_reliability.py --run-id reliability --baseline-run-id baseline
```
(Needs Phase 4 to have run on the baseline first.) ✅ Expect an R1-R4 table with PASS/FAIL.

### Phase 5 - scorecard and report
```bash
python scripts/run_phase5_report.py --run-id baseline --reliability-run-id reliability
```
Open the two files it prints: `Evaluation_Scorecard.xlsx` (open in Excel - formulas calculate on open) and
`report.html` (open in any browser).

### Rough size of a run (two ~1,000-word documents, OpenAI mini models)
| Phase | LLM calls | Time |
|---|---|---|
| 1 | ~70 | 5-20 min |
| 2 | ~350 | 15-30 min |
| 2b | ~80 | 5 min |
| 3 | ~540 answers | 10-15 min (4 parallel workers) |
| 4 / 4b | ~320 / ~420 judge calls | 5-10 min each |

Costs and times grow with the number of sections and questions. The golden-set answers by `gpt-4.1-mini`
cost about $0.07 in our run; the long-context reliability tests (up to 100K tokens each) are the largest item.

---

## 7. Where results are saved

| Folder | Contents |
|---|---|
| `data/processed/<doc>/` | Phase 0 chunks: `parents.jsonl`, `children.jsonl`, `structure_review.md` |
| `data/processed/alignment/` | Phase 1: `fact_pairs.jsonl`, `topic_map.json`, `alignment_review.md` |
| `data/golden/` | Phase 2: `golden.jsonl`, `rejected.jsonl`, `golden_review.md` |
| `data/reliability/` | Phase 2b: `variants.jsonl`, `filler.json`, `suite_review.md` |
| `data/vectorstore/` | Chroma stores (rebuilt automatically when chunks or embedding model change) |
| `outputs/runs/<run_id>/` | Phase 3: `retrieval.jsonl` (ranked chunk IDs), `generation.jsonl` (answers + citations) |
| `outputs/runs/<run_id>/eval/` | Phase 4: `summary.json`, `per_question.jsonl`, `scores.csv` |
| `outputs/runs/<run_id>/eval_reliability/` | Phase 4b: `summary.json`, `per_variant.jsonl` |
| `outputs/reports/<date>_<run_id>/` | Phase 5: `Evaluation_Scorecard.xlsx`, `report.html` |

---

## 8. Interruptions, failures and re-runs

* **Every LLM step saves each result immediately.** If a run stops (network, crash, Ctrl+C, quota), run the
  **same command again** - finished items are skipped.
* Items that failed after all retries are listed in `failed_*.jsonl` next to the results. Re-run only those:
  add `--retry-failed`.
* Start a phase from scratch: `--fresh` (Phases 1 and 2).
* Phases refuse to mix results made from different inputs or models, and tell you to use `--fresh`
  (Phases 1-2) or a new `--run-id` (Phase 3). This is deliberate.
* After changing documents: run Phase 0 again, then Phase 1 and 2 with `--fresh`, and new run ids.

---

## 9. Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `quota exhausted` / `insufficient_quota` / `429 ... per day` | Out of credit or free-tier limit. Add billing, then re-run the same command (it resumes). |
| `model ... not found` / `404` | Model name not available to your key - update `models:` in `config.yaml` (see step 5b). |
| `Phase 0 chunks or the models changed ... Re-run with --fresh` | Inputs changed since the last run - add `--fresh`. |
| `Run <id> was made with different settings/questions` | Use a new `--run-id`. |
| Structure review shows wrong sections | PDF without clear heading fonts: export the document as DOCX (with Heading styles) or Markdown and point `path` to it; or tune `ingestion.heading_min_size_delta` / `parent_heading_min_count`. |
| Phase 0 finds no text | Scanned PDF (images). Run OCR first (e.g. save as searchable PDF). |
| Warning about symlinks / HF_TOKEN (Windows, Phase 4) | Harmless (Hugging Face cache); the NLI model still loads. |
| `retrieval gate failed` | Retrieval too weak for reliable answers - check the weakest questions it prints; consider smaller `ingestion.child_tokens` or a larger `rag.context_k`. |
| Calls are very slow | Provider load. The default 90 s timeout retries stalled calls automatically; `llm.timeout_s` in config. |

---

## 10. Everything in one go (after configuring)

```bash
python scripts/run_phase0_ingest.py
python scripts/run_phase1_align.py
python scripts/run_phase2_golden.py
python scripts/run_phase2b_reliability_suite.py
python scripts/run_phase3_rag.py --retrieval-only
python scripts/run_phase3_rag.py --run-id baseline
python scripts/run_phase3_rag.py --questions data/reliability/variants.jsonl --run-id reliability
python scripts/run_phase4_evaluate.py --run-id baseline
python scripts/run_phase4b_reliability.py --run-id reliability --baseline-run-id baseline
python scripts/run_phase5_report.py --run-id baseline --reliability-run-id reliability
```
(Still review the Phase 0, 1 and 2 outputs before trusting the final scores.)
