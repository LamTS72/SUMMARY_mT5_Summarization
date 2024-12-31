class ConfigDataset():
    PATH_DATASET = "/kaggle/input/review/data"
    REVISION = None
    LANG1 = "en"
    LANG2 = "es"


class ConfigModel():
    BATCH_SIZE = 8
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 32
    MODEL_TOKENIZER = "google/mt5-small"
    MODEL_NAME = "google/mt5-small"
    TRAIN_SIZE = 0.9
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    METRICs = "rouge"
    PATH_TENSORBOARD = "runs/data_run"
    PATH_SAVE = "summary"
    NUM_WARMUP_STEPS = 0

class ConfigHelper():
    TOKEN_HF = "xxx"
    AUTHOR = "Chessmen"