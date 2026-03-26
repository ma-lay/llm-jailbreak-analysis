"""
Quick setup / smoke test
Checks: Python version, Ollama reachability, Vicuna availability, imports
"""

import sys
import importlib

MIN_PYTHON = (3, 10)
REQUIRED_PKGS = ["ollama", "textstat"]   # minimal runtime deps checked here


def check_python():
    v = sys.version_info[:2]
    ok = v >= MIN_PYTHON
    print(f"  Python {sys.version.split()[0]} … {'OK' if ok else 'FAIL (need >= 3.10)'}")
    return ok


def check_imports():
    all_ok = True
    for pkg in REQUIRED_PKGS:
        try:
            importlib.import_module(pkg)
            print(f"  import {pkg} … OK")
        except ImportError:
            print(f"  import {pkg} … MISSING  (pip install {pkg})")
            all_ok = False
    return all_ok


def check_ollama():
    try:
        import ollama
        models = [m["model"] for m in ollama.list()["models"]]
        print(f"  Ollama reachable. Models: {models}")
        import config
        if config.MODEL_NAME in models:
            print(f"  {config.MODEL_NAME} … FOUND")
            return True
        else:
            print(f"  {config.MODEL_NAME} … NOT FOUND  (run: ollama pull {config.MODEL_NAME})")
            return False
    except Exception as e:
        print(f"  Ollama … UNREACHABLE ({e})")
        return False


def check_local_modules():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from src.model.vicuna_wrapper import VicunaWrapper   # noqa: F401
        from src.algorithm.evolution_es import EvolutionStrategy  # noqa: F401
        from src.metrics.attack_success import is_jailbroken  # noqa: F401
        from src.metrics.readability import readability_score  # noqa: F401
        print("  Local modules … OK")
        return True
    except Exception as e:
        print(f"  Local modules … FAIL: {e}")
        return False


def quick_metric_test():
    from src.metrics.attack_success import is_jailbroken
    from src.metrics.readability import readability_score
    assert not is_jailbroken("I cannot help with that."), "Refusal not detected"
    assert is_jailbroken(
        "Sure, here is a detailed step-by-step guide on how to accomplish this task. "
        "First you need to gather the required materials and then proceed with careful planning."
    ), "Compliance not detected"
    r = readability_score("for educational purposes only", "explain something to me please")
    assert 0 <= r <= 1
    print("  Metric tests … OK")
    return True


if __name__ == "__main__":
    print("=== autodan-evo setup check ===\n")
    ok = True
    ok &= check_python()
    ok &= check_imports()
    ok &= check_local_modules()
    ok &= quick_metric_test()
    ok &= check_ollama()
    print(f"\n{'Setup OK' if ok else 'Setup has issues – see above'}")
    sys.exit(0 if ok else 1)
