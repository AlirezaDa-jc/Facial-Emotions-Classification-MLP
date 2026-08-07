import os
import torch

# Paths
DATASET_PATH = "./dataset/"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Fixed settings (shared across all experiments)
VAL_SPLIT = 0.2
SEED = 42
INPUT_SIZE = 48 * 48  # 2304
NUM_CLASSES = 7
DROPOUT_RATE = 0.5
EPOCHS = 20
NUM_WORKERS = 0 # Windows = 0

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output locations
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# ---------------------------------------------------------------------------
# Named experiment configs.
# Each key is an experiment name; each value is the set of hyperparameters
# that differ from the baseline. Run one by name in main.py.
# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline": {
        "hidden_sizes": (512,),
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 64,
    },

    # --- Number of hidden layers ---
    "deeper_2layer": {
        "hidden_sizes": (512, 256),
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 64,
    },
    "deeper_3layer": {
        "hidden_sizes": (512, 256, 128),
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 64,
    },

    # --- Neurons per layer (single layer, varying width) ---
    "narrow_128": {
        "hidden_sizes": (128,),
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 64,
    },
    "wide_1024": {
        "hidden_sizes": (1024,),
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 64,
    },

    # --- Activation function ---
    "sigmoid_activation": {
        "hidden_sizes": (512,),
        "activation": "sigmoid",
        "learning_rate": 0.001,
        "batch_size": 64,
    },

    # --- Learning rate ---
    "lr_high_0_01": {
        "hidden_sizes": (512,),
        "activation": "relu",
        "learning_rate": 0.01,
        "batch_size": 64,
    },
    "lr_low_0_0001": {
        "hidden_sizes": (512,),
        "activation": "relu",
        "learning_rate": 0.0001,
        "batch_size": 64,
    },

    # --- Batch size ---
    "batch_32": {
        "hidden_sizes": (512,),
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 32,
    },
    "batch_128": {
        "hidden_sizes": (512,),
        "activation": "relu",
        "learning_rate": 0.001,
        "batch_size": 128,
    },
}