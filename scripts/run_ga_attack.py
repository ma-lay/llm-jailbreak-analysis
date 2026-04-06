"""GA runner that reuses the existing attack pipeline."""

from scripts import run_attack as base_runner
from src.algorithm.ga import GeneticAlgorithm


def main() -> None:
    original_class = base_runner.ALGORITHM_CLASS
    original_name = base_runner.ALGORITHM_NAME
    try:
        base_runner.ALGORITHM_CLASS = GeneticAlgorithm
        base_runner.ALGORITHM_NAME = "GA"
        base_runner.main()
    finally:
        base_runner.ALGORITHM_CLASS = original_class
        base_runner.ALGORITHM_NAME = original_name


if __name__ == "__main__":
    main()
