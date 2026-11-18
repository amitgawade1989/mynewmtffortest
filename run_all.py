"""Run the whole pipeline (Phase 0 -> Phase 5) with one command.

  python run_all.py                       # every step, run IDs baseline_sample / reliability_sample / safety_sample
  python run_all.py --tag v2              # run IDs baseline_v2 / reliability_v2 / safety_v2
  python run_all.py --install             # pip install -r requirements-run.txt first
  python run_all.py --from-step 7         # resume from step 7 (see --list)
  python run_all.py --only 12             # run a single step
  python run_all.py --retry-failed        # pass --retry-failed to every step that supports it
  python run_all.py --list                # show the numbered steps and exit

Every step is resumable, so re-running after a failure continues where it stopped.
Stops at the first failing step and prints the command to resume. Output is also written to
outputs/logs/run_all_<timestamp>.log. Needs OPENAI_API_KEY (or GOOGLE_API_KEY) in .env.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RETRY = {"run_phase1_align.py", "run_phase2_golden.py", "run_phase2b_reliability_suite.py", "run_phase2c_safety_suite.py",
         "run_phase3_rag.py", "run_phase4_evaluate.py", "run_phase4b_reliability.py", "run_phase4c_safety.py"}


def steps(tag):
    base, rel, saf = f"baseline_{tag}", f"reliability_{tag}", f"safety_{tag}"
    return [
        ("Phase 0  ingest documents", ["run_phase0_ingest.py"]),
        ("Check embeddings (no API, preflight)", ["check_embeddings.py"]),   # needs Phase 0 chunks
        ("Phase 1  align handbooks", ["run_phase1_align.py"]),
        ("Phase 2  golden question set", ["run_phase2_golden.py"]),
        ("Phase 2b reliability suite", ["run_phase2b_reliability_suite.py"]),
        ("Phase 2c safety suite", ["run_phase2c_safety_suite.py"]),
        ("Phase 3  RAG on golden set", ["run_phase3_rag.py", "--run-id", base]),
        ("Phase 4  evaluate golden run", ["run_phase4_evaluate.py", "--run-id", base]),
        ("Phase 3  RAG on reliability suite", ["run_phase3_rag.py", "--questions", "data/reliability/variants.jsonl", "--run-id", rel]),
        ("Phase 4b evaluate reliability", ["run_phase4b_reliability.py", "--run-id", rel, "--baseline-run-id", base]),
        ("Phase 3  RAG on safety suite", ["run_phase3_rag.py", "--questions", "data/safety/variants.jsonl", "--run-id", saf]),
        ("Phase 4c evaluate safety", ["run_phase4c_safety.py", "--run-id", saf, "--baseline-run-id", base]),
        ("Phase 5  scorecard + report", ["run_phase5_report.py", "--run-id", base, "--reliability-run-id", rel, "--safety-run-id", saf]),
    ]


def run(cmd, log, env):
    """Run one command, echo its output live and append it to the log. Returns the exit code."""
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return p.wait()


def main():
    ap = argparse.ArgumentParser(description="Run the full evaluation pipeline")
    ap.add_argument("--tag", default="sample", help="suffix of the run IDs (default: sample)")
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--install", action="store_true", help="pip install -r requirements-run.txt first")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--config", default=None, help="config file passed to every step")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    plan = steps(args.tag)
    if args.list:
        for i, (name, cmd) in enumerate(plan, 1):
            print(f"{i:>2}. {name:<36} python scripts/{' '.join(cmd)}")
        return
    if not (ROOT / ".env").exists():
        sys.exit("No .env file - copy .env.example to .env and put your API key in it.")

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"run_all_{datetime.now():%Y%m%d_%H%M%S}.log"

    if args.install:
        if run([sys.executable, "-m", "pip", "install", "-r", "requirements-run.txt"], log, env):
            sys.exit("pip install failed - see the log above.")

    selected = [(i, s) for i, s in enumerate(plan, 1)
                if (i == args.only if args.only else i >= args.from_step)]
    for i, (name, cmd) in selected:
        full = [sys.executable, str(ROOT / "scripts" / cmd[0]), *cmd[1:]]
        if args.retry_failed and cmd[0] in RETRY:
            full.append("--retry-failed")
        if args.config:
            full += ["--config", args.config]
        print(f"\n{'=' * 78}\n[{i}/{len(plan)}] {name}\n{'=' * 78}", flush=True)
        start = datetime.now()
        code = run(full, log, env)
        if code != 0:
            print(f"\nStep {i} failed (exit {code}). Fix the problem, then resume with:\n"
                  f"  python run_all.py --tag {args.tag} --from-step {i}\nLog: {log}")
            sys.exit(code)
        print(f"[{i}/{len(plan)}] done in {(datetime.now() - start).seconds} s")
    print(f"\nAll steps finished. Report: outputs/reports/ (newest folder)\nLog: {log}")


if __name__ == "__main__":
    main()
