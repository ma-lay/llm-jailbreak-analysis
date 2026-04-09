"""Global configuration for AutoDAN-Evo"""


# Supported examples: "vicuna", "mistral", "llama2"
MODEL_NAME       = "vicuna:7b"
OLLAMA_BASE_URL  = "http://localhost:11434"
MAX_TOKENS       = 150                # Max response tokens
TEMPERATURE      = 0.7

# Attack optimizer: "es" (default), "ga", or "pso"
ATTACK_ALGORITHM = "es"

PARALLEL_WORKERS   = 4
STAGNATION_LIMIT   = 10
DIVERSITY_THRESHOLD = 0.3
IMMIGRANT_FRACTION  = 0.2
GENERATIONS        = 15
MUTATION_RATE      = 0.4
EARLY_STOP_THRESHOLD = 0.8

ES_MU            = 6
ES_LAMBDA        = 8

MAX_SUFFIX_RATIO = 0.5
RESULTS_DIR      = "outputs"
