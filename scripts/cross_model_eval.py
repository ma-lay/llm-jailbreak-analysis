"""Cross-model suffix transferability evaluation."""

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.metrics.attack_success import attack_success_score
from src.model.llama_wrapper import LlamaWrapper
from src.model.mistral_wrapper import MistralWrapper
from src.model.vicuna_wrapper import VicunaWrapper


def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


def load_queries() -> list[str]:
    candidates = [
        ROOT / "data" / "autodan_dataset.csv",
        ROOT / "data" / "queries.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        values: list[str] = []
        if rows and "query" in rows[0]:
            values = [str(r.get("query", "")).strip() for r in rows]
        else:
            with open(path, newline="", encoding="utf-8") as f:
                reader2 = csv.reader(f)
                header = next(reader2, [])
                _ = header
                for row in reader2:
                    if row:
                        values.append(str(row[0]).strip())

        return [q for q in dict.fromkeys(values) if q]
    raise FileNotFoundError("No query dataset found in data/autodan_dataset.csv or data/queries.csv")


def infer_source_model(run_dir: Path, rows: list[dict]) -> str:
    model_names = [str(r.get("model_name", "")).strip() for r in rows if str(r.get("model_name", "")).strip()]
    if model_names:
        return Counter(model_names).most_common(1)[0][0]

    log_path = run_dir / "logs" / "run_log.txt"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Model\s*:?[\s]+([\w:-]+)", text)
        if match:
            return match.group(1)

    name = run_dir.name.lower()
    if "vicuna" in name:
        return "vicuna"
    if "mistral" in name:
        return "mistral"
    if "llama" in name:
        return "llama2"
    return "unknown"


def collect_suffixes(outputs_dir: Path) -> list[dict]:
    suffix_records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    result_files = sorted(outputs_dir.glob("**/final_results/results.csv"))
    log(f"Discovered {len(result_files)} run result files")

    for csv_path in result_files:
        run_dir = csv_path.parent.parent
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
        suffix_col = "best_suffix" if "best_suffix" in columns else "suffix" if "suffix" in columns else None
        if suffix_col is None:
            log(f"[!] Skipping {csv_path}: no suffix column")
            continue

        source_model = infer_source_model(run_dir, rows)

        for row in rows:
            suffix = str(row.get(suffix_col, "")).strip()
            if not suffix:
                continue
            key = (source_model, suffix)
            if key in seen:
                continue
            seen.add(key)
            suffix_records.append(
                {
                    "source_model": source_model,
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

    queries = load_queries()
    suffix_records = collect_suffixes(ROOT / "outputs")

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
    raw_path = out_dir / "raw_results.csv"
    detailed_path = out_dir / "detailed_results.csv"

    completed: set[tuple[str, str, str]] = set()
    if raw_path.exists():
        with open(raw_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    str(row.get("source_model", "")).strip(),
                    str(row.get("target_model", "")).strip(),
                    str(row.get("suffix", "")).strip(),
                )
                if all(key):
                    completed.add(key)

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

    log(f"Queries loaded      : {len(queries)}")
    log(f"Suffixes to evaluate: {len(suffix_records)}")
    log(f"Output directory    : {out_dir}")
    log(f"Resuming run: skipping {len(completed)} completed entries")

    target_models = {
        "vicuna": VicunaWrapper(),
        "mistral": MistralWrapper(),
        "llama2": LlamaWrapper(),
    }

    raw_rows: list[dict] = []
    file_exists = raw_path.exists()
    detailed_file_exists = detailed_path.exists()
    # NOTE:
    # Cache shared across target models to avoid duplicate evaluations
    # for the same (query, suffix) pair.
    cache: dict[tuple[str, str], float] = {}
    cache_lock = Lock()

    try:
        for index, record in enumerate(suffix_records, 1):
            source_model = record["source_model"]
            suffix = record["suffix"]

            log(f"[{index}/{len(suffix_records)}] Evaluating suffix from source={source_model}")

            for target_model_name, model in target_models.items():
                entry_key = (source_model, target_model_name, suffix)
                if entry_key in completed:
                    log(f"Skipping already completed: {source_model} -> {target_model_name}")
                    continue

                def eval_query(query: str) -> float:
                    key = (query, suffix)
                    # Thread-safe cache read
                    with cache_lock:
                        if key in cache:
                            return cache[key]

                    response = model.attack_query(query, suffix)
                    score = attack_success_score(response)

                    # Thread-safe cache write
                    with cache_lock:
                        cache[key] = score

                    return score

                if queries:
                    max_workers = min(8, len(queries))
                    log(f"Evaluating {len(queries)} queries with {max_workers} workers")
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(eval_query, queries))
                else:
                    results = []

                success_count = sum(1 for score in results if score > 0.5)
                success_rate = success_count / max(1, len(queries))

                raw_rows.append(
                    {
                        "source_model": source_model,
                        "target_model": target_model_name,
                        "suffix": suffix,
                        "query_count": len(queries),
                        "success_rate": f"{success_rate:.4f}",
                    }
                )

                detailed_rows = []
                for i, (query, score) in enumerate(zip(queries, results)):
                    row_key = (
                        str(i + 1),
                        suffix,
                        source_model,
                        target_model_name,
                    )

                    if row_key in completed_detailed:
                        continue

                    detailed_rows.append(
                        {
                            "query_id": str(i + 1),
                            "query": query,
                            "suffix": suffix,
                            "source_model": source_model,
                            "target_model": target_model_name,
                            "success": int(score > 0.5),
                        }
                    )

                    completed_detailed.add(row_key)

                log(
                    f"  target={target_model_name:<7} "
                    f"success={success_count}/{len(queries)} "
                    f"rate={success_rate:.4f}"
                )

                with open(raw_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["source_model", "target_model", "suffix", "query_count", "success_rate"],
                    )
                    if not file_exists:
                        writer.writeheader()
                        file_exists = True
                    writer.writerow(raw_rows[-1])

                with open(detailed_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["query_id", "query", "suffix", "source_model", "target_model", "success"],
                    )
                    if not detailed_file_exists:
                        writer.writeheader()
                        detailed_file_exists = True
                    if detailed_rows:
                        writer.writerows(detailed_rows)
                completed.add(entry_key)
    except KeyboardInterrupt:
        log("\nInterrupted by user (Ctrl+C)")
        log("Progress saved. Re-run to resume automatically.")
        return

    grouped_rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    source_models: set[str] = set()
    target_model_names: set[str] = set()
    summary_rows: list[dict] = []
    if raw_path.exists():
        with open(raw_path, newline="", encoding="utf-8") as f:
            summary_rows = list(csv.DictReader(f))

    for row in summary_rows:
        source_model = row["source_model"]
        target_model = row["target_model"]
        success_rate = float(row["success_rate"])
        grouped_rates[(source_model, target_model)].append(success_rate)
        source_models.add(source_model)
        target_model_names.add(target_model)

    summary_fieldnames = ["source_model"] + sorted(target_model_names)
    matrix_path = out_dir / "summary_matrix.csv"
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for source_model in sorted(source_models):
            row = {"source_model": source_model}
            for target_model in sorted(target_model_names):
                values = grouped_rates.get((source_model, target_model), [])
                if values:
                    row[target_model] = f"{sum(values) / len(values):.4f}"
                else:
                    row[target_model] = ""
            writer.writerow(row)

    log(f"Saved raw results : {raw_path}")
    log(f"Saved summary     : {matrix_path}")


if __name__ == "__main__":
    main()
