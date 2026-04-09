"""PSO runner that reuses the existing attack pipeline."""

import run_attack as base_runner
from src.algorithm.pso import ParticleSwarmOptimization


def main() -> None:
    original_class = base_runner.ALGORITHM_CLASS
    original_name = base_runner.ALGORITHM_NAME
    try:
        base_runner.ALGORITHM_CLASS = ParticleSwarmOptimization
        base_runner.ALGORITHM_NAME = "PSO"
        base_runner.main()
    finally:
        base_runner.ALGORITHM_CLASS = original_class
        base_runner.ALGORITHM_NAME = original_name


if __name__ == "__main__":
    main()