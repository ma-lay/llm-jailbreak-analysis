"""Genetic Algorithm variant for fair comparison with ES."""

import random
import time
from pathlib import Path
from typing import List

import config
from src.algorithm.evolution_es import Candidate, EvolutionStrategy


def crossover(parent1: str, parent2: str) -> str:
    """Combine two parent suffixes by midpoint crossover."""
    words1 = parent1.split()
    words2 = parent2.split()

    if len(words1) < 2 or len(words2) < 2:
        return parent1

    split1 = len(words1) // 2
    split2 = len(words2) // 2

    child_words = words1[:split1] + words2[split2:]
    return " ".join(child_words)


class GeneticAlgorithm(EvolutionStrategy):
    """GA implementation reusing ES evaluation/mutation/fitness for fair comparison."""

    def run(
        self,
        query: str,
        expected_output: str = "",
        query_id: str | None = None,
        history_dir: Path | None = None,
    ) -> List[Candidate]:
        start = time.time()
        self.history = []
        self.seen_suffixes = set()
        self.current_mutation_rate = self.base_mutation_rate

        parents = self._init_pop(query)
        self._evaluate(parents, query, expected_output)
        parents.sort(key=lambda c: c.fitness, reverse=True)

        if self.verbose:
            print(
                f"\n[GA] Query: {query[:80]}\n"
                f"     mu={self.mu}, lambda={self.lambda_}, generations={self.generations}"
            )

        stag_cnt = 0
        prev_best = -float("inf")

        for gen in range(1, self.generations + 1):
            offspring: List[Candidate] = []

            for _ in range(self.lambda_):
                parent1, parent2 = random.sample(parents, 2)

                crossed = crossover(parent1.suffix, parent2.suffix)
                pseudo_parent = Candidate(
                    suffix=crossed,
                    read_score=(parent1.read_score + parent2.read_score) / 2.0,
                    generation=gen,
                    mutation_type="crossover",
                )

                child_text, mutation_type = self._mutate(pseudo_parent, query)
                child_mutation_type = f"crossover+{mutation_type}"

                offspring.append(
                    Candidate(
                        suffix=child_text,
                        generation=gen,
                        mutation_type=child_mutation_type,
                    )
                )

            self._evaluate(offspring, query, expected_output)

            combined_population = parents + offspring
            combined_population.sort(key=lambda c: c.fitness, reverse=True)
            parents = combined_population[: self.mu]

            best = parents[0]
            if best.fitness > prev_best + 1e-6:
                stag_cnt = 0
                prev_best = best.fitness
            else:
                stag_cnt += 1

            div = len(set(c.suffix for c in parents)) / max(1, len(parents))

            if stag_cnt >= 2:
                self.current_mutation_rate = min(0.9, self.current_mutation_rate + 0.1)
            elif div < 0.5:
                self.current_mutation_rate = min(0.8, self.current_mutation_rate + 0.05)
            else:
                self.current_mutation_rate = max(
                    self.base_mutation_rate,
                    self.current_mutation_rate - 0.05,
                )

            if div < config.DIVERSITY_THRESHOLD:
                parents = self._inject_immigrants(parents, query, gen)
                self._evaluate(parents, query, expected_output)
                parents.sort(key=lambda c: c.fitness, reverse=True)
                best = parents[0]

            self._record_gen(parents, gen, self.current_mutation_rate, div)

            if self.verbose:
                n_jb = sum(1 for c in parents if c.attack_score > 0.5)
                print(
                    f"  Gen {gen:3d} | best={best.fitness:.3f} "
                    f"(asr={best.attack_score:.1f} "
                    f"align={best.alignment_score:.2f} "
                    f"len={best.length_score:.2f} "
                    f"read={best.read_score:.2f}) | "
                    f"jailbroken={n_jb}/{len(parents)} | "
                    f"div={div:.2f} stag={stag_cnt}"
                )

            if best.fitness >= config.EARLY_STOP_THRESHOLD and best.attack_score > 0.5:
                if self.verbose:
                    print("  [GA] Early stopping - threshold reached.")
                break

            if stag_cnt >= config.STAGNATION_LIMIT:
                if self.verbose:
                    print(f"  [GA] Stagnation at gen {gen}.")
                break

        self.query_runtime_s = time.time() - start
        if query_id:
            self._persist(query_id, "ga", history_dir)

        return parents