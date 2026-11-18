# Run steps

```powershell
cd "<path-to>\Document comparision"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-run.txt
Copy-Item .env.example .env
notepad .env
$env:PYTHONIOENCODING = "utf-8"
python scripts/run_phase0_ingest.py
python scripts/check_embeddings.py
python scripts/run_phase1_align.py
python scripts/run_phase2_golden.py
python scripts/run_phase2b_reliability_suite.py
python scripts/run_phase2c_safety_suite.py
python scripts/run_phase3_rag.py --run-id baseline_sample
python scripts/run_phase4_evaluate.py --run-id baseline_sample
python scripts/run_phase3_rag.py --questions data/reliability/variants.jsonl --run-id reliability_sample
python scripts/run_phase4b_reliability.py --run-id reliability_sample --baseline-run-id baseline_sample
python scripts/run_phase3_rag.py --questions data/safety/variants.jsonl --run-id safety_sample
python scripts/run_phase4c_safety.py --run-id safety_sample --baseline-run-id baseline_sample
python scripts/run_phase5_report.py --run-id baseline_sample --reliability-run-id reliability_sample --safety-run-id safety_sample
```
