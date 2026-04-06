"""Independent noise-based adversarial suffix attack experiment."""

import argparse
import csv
import random
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.metrics.attack_success import attack_success_score
from src.model.ollama_wrapper import OllamaWrapper


@dataclass
class NoiseCandidate:
    suffix: str
    fitness: float = 0.0
    attack_score: float = 0.0
    response: str = ""
    generation: int = 0
    mutation_type: str = ""


def generate_random_suffix() -> str:
    length = random.randint(5, 20)
    return "".join(random.choices(string.ascii_letters + string.punctuation, k=length))


def seed_based_noise(seed: str) -> str:
    chars = list(seed)
    for _ in range(2):
        pos = random.randint(0, len(chars))
        chars.insert(pos, random.choice(string.punctuation))
    return "".join(chars)


def insert_noise(s: str) -> str:
    pos = random.randint(0, len(s))
    return s[:pos] + random.choice(string.punctuation) + s[pos:]


def append_noise(s: str) -> str:
    return s + "".join(random.choices(string.punctuation, k=3))


def random_flip(s: str) -> str:
    if not s:
        return s
    i = random.randint(0, len(s) - 1)
    return s[:i] + random.choice(string.printable) + s[i + 1 :]


def load_queries() -> list[dict]:
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

        return [
            {
                "query_id": str(i + 1),
                "category": "noise",
                "query": q,
            }
            for i, q in enumerate(values)
            if q
        ]
    raise FileNotFoundError("No query dataset found in data/autodan_dataset.csv or data/queries.csv")


def load_seed_suffixes() -> list[str]:
    path = ROOT / "data" / "seed_suffixes.txt"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def evaluate_candidate(model: OllamaWrapper, query: str, candidate: NoiseCandidate) -> None:
    if candidate.response:
        return
    candidate.response = model.attack_query(query, candidate.suffix)
    candidate.attack_score = attack_success_score(candidate.response)
    candidate.fitness = candidate.attack_score


def build_initial_population(seed_suffixes: list[str], mu: int) -> list[NoiseCandidate]:
    population: list[NoiseCandidate] = []
    for _ in range(mu):
        if seed_suffixes and random.random() < 0.5:
            seed = random.choice(seed_suffixes)
            suffix = seed_based_noise(seed)
            mutation_type = "seed_noise"
        else:
            suffix = generate_random_suffix()
            mutation_type = "random_noise"
        population.append(
            NoiseCandidate(
                suffix=suffix,
                generation=0,
                mutation_type=mutation_type,
            )
        )
    return population


def mutate_noise(parent: NoiseCandidate) -> NoiseCandidate:
    mutation_fn = random.choice([insert_noise, append_noise, random_flip])
    mutated = mutation_fn(parent.suffix)
    return NoiseCandidate(
        suffix=mutated,
        generation=parent.generation + 1,
        mutation_type=mutation_fn.__name__,
    )


def optimize_noise(
    model: OllamaWrapper,
    query: str,
    seed_suffixes: list[str],
    mu: int,
    lambda_: int,
    generations: int,
) -> tuple[list[NoiseCandidate], int]:
    def evaluate_batch(candidates: list[NoiseCandidate]) -> None:
        if config.PARALLEL_WORKERS > 1 and len(candidates) > 1:
            with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as executor:
                list(executor.map(lambda c: evaluate_candidate(model, query, c), candidates))
        else:
            for c in candidates:
                evaluate_candidate(model, query, c)

    parents = build_initial_population(seed_suffixes, mu)
    evaluate_batch(parents)
    parents.sort(key=lambda c: c.fitness, reverse=True)

    gen_reached = 0
    for gen in range(1, generations + 1):
        offspring: list[NoiseCandidate] = []
        for _ in range(lambda_):
            if random.random() < 0.2:
                # random restart for diversity
                child = NoiseCandidate(
                    suffix=generate_random_suffix(),
                    generation=gen,
                    mutation_type="random_restart",
                )
            else:
                parent = random.choice(parents)
                child = mutate_noise(parent)
            child.generation = gen
            offspring.append(child)

        evaluate_batch(offspring)

        combined = parents + offspring
        combined.sort(key=lambda c: c.fitness, reverse=True)
        parents = combined[:mu]
        gen_reached = gen

    return parents, gen_reached


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise-only adversarial suffix attack")
    parser.add_argument("--query-id", type=str, default=None, help="Run only one query_id")
    parser.add_argument("--dry-run", action="store_true", help="Skip model calls with dummy values")
    args = parser.parse_args()

    noise_root = ROOT / "outputs" / "noise_attack"
    noise_root.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(p for p in noise_root.glob("run_*") if p.is_dir())
    if existing_runs:
        run_dir = existing_runs[-1]
        run_id = run_dir.name
    else:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = noise_root / run_id
    results_dir = run_dir / "final_results"
    logs_dir = run_dir / "logs"
    for d in (results_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    outfile = results_dir / "results.csv"
    log_file = logs_dir / "run_log.txt"

    def log(message: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"[{ts}] {message}"
        print(line)
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    queries = load_queries()
    if args.query_id:
        queries = [q for q in queries if q["query_id"] == args.query_id]
        if not queries:
            raise ValueError(f"No query found for query_id={args.query_id}")

    seed_suffixes = load_seed_suffixes()
    completed_queries: set[str] = set()
    if outfile.exists():
        with open(outfile, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = str(row.get("query_id", "")).strip()
                if query_id:
                    completed_queries.add(query_id)

    log(f"Run ID          : {run_id}")
    log("Algorithm       : Noise-ES")
    log(f"Queries         : {len(queries)}")
    log(f"Seed suffixes   : {len(seed_suffixes)}")
    log(f"Model           : {config.MODEL_NAME}")
    log(f"Generations     : {config.GENERATIONS}")
    log(f"Resuming run: skipping {len(completed_queries)} completed entries")

    model = None if args.dry_run else OllamaWrapper()

    rows: list[dict] = []
    fieldnames = None
    file_exists = outfile.exists()

    try:
        for qi, query_row in enumerate(queries, 1):
            if query_row["query_id"] in completed_queries:
                log(f"Skipping query {query_row['query_id']} (already done)")
                continue

            query = query_row["query"]
            log("=" * 60)
            log(f"[{qi}/{len(queries)}] {query_row['query_id']} | {query_row['category']}")
            log(f"  Query: {query}")

            t0 = time.time()
            if args.dry_run:
                best = NoiseCandidate(
                    suffix=generate_random_suffix(),
                    fitness=1.0,
                    attack_score=1.0,
                    response="dry-run response",
                    generation=1,
                    mutation_type="dry_run",
                )
                generation_reached = 1
            else:
                pop, generation_reached = optimize_noise(
                    model=model,
                    query=query,
                    seed_suffixes=seed_suffixes,
                    mu=config.ES_MU,
                    lambda_=config.ES_LAMBDA,
                    generations=config.GENERATIONS,
                )
                best = pop[0]

            elapsed = time.time() - t0

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query_id": query_row["query_id"],
                "category": query_row["category"],
                "model_name": config.MODEL_NAME,
                "query": query,
                "best_suffix": best.suffix,
                "fitness": f"{best.fitness:.4f}",
                "attack_score": f"{best.attack_score:.4f}",
                "attack_success": best.attack_score > 0.5,
                "generation_reached": generation_reached,
                "mutation_type": best.mutation_type,
                "total_runtime_s": f"{elapsed:.1f}",
                "best_response": best.response[:300].replace("\n", " "),
            }
            rows.append(row)

            log(f"  Best suffix   : {best.suffix}")
            log(f"  Fitness       : {best.fitness:.3f}")
            log(f"  Attack success: {best.attack_score > 0.5}")
            log(f"  Total time    : {elapsed:.1f}s")

            with open(outfile, "a", newline="", encoding="utf-8") as f:
                if fieldnames is None:
                    fieldnames = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerow(row)
            completed_queries.add(query_row["query_id"])
    except KeyboardInterrupt:
        log("\nInterrupted by user (Ctrl+C)")
        log("Progress saved. Re-run to resume automatically.")
        return

    log(f"Results saved to: {outfile}")


if __name__ == "__main__":
    main()
