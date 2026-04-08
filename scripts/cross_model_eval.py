"""Cross-model suffix transferability evaluation."""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Each model maps to ONE specific run directory.
# This ensures consistent and reproducible comparisons.
MODEL_RUN_PATHS = {
    "vicuna": ROOT / "outputs" / "vicuna_run_name_here",
    "mistral": ROOT / "outputs" / "mistral_run_name_here",
    "llama2": ROOT / "outputs" / "llama_run_name_here",
}

from src.metrics.attack_success import attack_success_score
from src.model.llama_wrapper import LlamaWrapper
from src.model.mistral_wrapper import MistralWrapper
from src.model.vicuna_wrapper import VicunaWrapper


def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


def collect_suffixes() -> list[dict]:
    suffix_records: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for model_name, run_dir in MODEL_RUN_PATHS.items():
        csv_path = run_dir / "final_results" / "results.csv"

        if not csv_path.exists():
            log(f"[!] Missing results file for {model_name}: {csv_path}")
            continue

        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as exc:
            log(f"[!] Skipping unreadable file {csv_path}: {exc}")
            continue

        if not rows:
            continue

        columns = set(rows[0].keys())
        suffix_col = "best_suffix" if "best_suffix" in columns else "suffix"

        for row in rows:
            suffix = str(row.get(suffix_col, "")).strip()
            query_id = str(row.get("query_id", "")).strip()
            query = str(row.get("query", "")).strip()
            if not suffix or not query_id or not query:
                continue

            key = (model_name, query_id, suffix)
            if key in seen:
                continue
            seen.add(key)
            suffix_records.append(
                {
                    "source_model": model_name,
                    "query_id": query_id,
                    "query": query,
                    "suffix": suffix,
                    "run_id": run_dir.name,
                }
            )

    return suffix_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model transferability evaluation")
    parser.add_argument(
        "--max-suffixes",
        type=int,
        default=0,
        help="Optional limit on number of suffixes to evaluate (0 = all)",
    )
    args = parser.parse_args()

    for model_name, path in MODEL_RUN_PATHS.items():
        log(f"Using run for {model_name}: {path}")

    suffix_records = collect_suffixes()

    if args.max_suffixes > 0:
        suffix_records = suffix_records[: args.max_suffixes]

    if not suffix_records:
        raise ValueError("No suffixes found for cross-model evaluation")

    eval_root = ROOT / "outputs" / "cross_model_eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(p for p in eval_root.glob("run_*") if p.is_dir())
    if existing_runs:
        out_dir = existing_runs[-1]
    else:
        out_dir = eval_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = out_dir / "detailed_results.csv"

    completed_detailed: set[tuple[str, str, str, str]] = set()
    if detailed_path.exists():
        with open(detailed_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    str(row.get("query_id", "")).strip(),
                    str(row.get("suffix", "")).strip(),
                    str(row.get("source_model", "")).strip(),
                    str(row.get("target_model", "")).strip(),
                )
                if all(key):
                    completed_detailed.add(key)

    log(f"Suffixes to evaluate: {len(suffix_records)}")
    log(f"Output directory    : {out_dir}")
    log(f"Resuming run: skipping {len(completed_detailed)} completed entries")

    target_models = {
        "vicuna": VicunaWrapper(),
        "mistral": MistralWrapper(),
        "llama2": LlamaWrapper(),
    }

    detailed_file_exists = detailed_path.exists()
    # NOTE:
    # Cache shared across target models to avoid duplicate evaluations
    # for the same (query, suffix) pair.
    cache: dict[tuple[str, str], float] = {}
    cache_lock = Lock()
    completed_detailed_lock = Lock()

    try:
        tasks = [
            (record, target_model_name)
            for record in suffix_records
            for target_model_name in target_models
        ]
        if tasks:
            max_workers = min(8, len(tasks))
            log(f"Evaluating {len(tasks)} tasks with {max_workers} workers")

            def run_task(task: tuple[dict, str]) -> dict | None:
                record, target_model_name = task
                source_model = record["source_model"]
                query_id = record["query_id"]
                query = record["query"]
                suffix = record["suffix"]
                entry_key = (query_id, suffix, source_model, target_model_name)
                with completed_detailed_lock:
                    if entry_key in completed_detailed:
                        return None

                model = target_models[target_model_name]

                key = (query, suffix, target_model_name)
                with cache_lock:
                    cached_score = cache.get(key)

                if cached_score is None:
                    response = model.attack_query(query, suffix)
                    score = attack_success_score(response)
                    with cache_lock:
                        cache[key] = score
                else:
                    score = cached_score

                with completed_detailed_lock:
                    completed_detailed.add(entry_key)
                return {
                    "query_id": query_id,
                    "query": query,
                    "suffix": suffix,
                    "source_model": source_model,
                    "target_model": target_model_name,
                    "success": int(score > 0.5),
                }

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for row in executor.map(run_task, tasks):
                    if row is None:
                        continue
                    with open(detailed_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["query_id", "query", "suffix", "source_model", "target_model", "success"],
                        )
                        if not detailed_file_exists:
                            writer.writeheader()
                            detailed_file_exists = True
                        writer.writerow(row)
    except KeyboardInterrupt:
        log("\nInterrupted by user (Ctrl+C)")
        log("Progress saved. Re-run to resume automatically.")
        return

    log(f"Saved detailed results : {detailed_path}")


if __name__ == "__main__":
    main()
