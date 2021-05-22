""" Relevant path variables and their associated absolute paths"""
from os import path


# # Path to project directory
# PROJECT_DIR_PATH = path.dirname(path.dirname(path.abspath(__file__)))

# # Train and test csv files
# TRAIN_CSV_FILE = path.join(PROJECT_DIR_PATH, "data", "training.1600000.processed.noemoticon.csv")
# TEST_CSV_FILE= path.join(PROJECT_DIR_PATH, "data", "testdata.manual.2009.06.14.csv")

# # Processed train/test data
# TRAIN_ALL_PROCESSED = path.join(PROJECT_DIR_PATH, "data", "train_all_processed.pickle")
# TEST_ALL_PROCESSED = path.join(PROJECT_DIR_PATH, "data", "test_all_processed.pickle")

# # Path to trained models directory
# TRAINED_MODELS = path.join(PROJECT_DIR_PATH, "trained_models")

# # Name of directory of best performing model so far (stored in trained models folder)
# NAME_DIR_BEST_MODEL = "bert_sentiment140_27008records_0.83acc_10_39_41"
# # Path to best performing model so far
# BEST_MODEL = path.join(TRAINED_MODELS ,NAME_DIR_BEST_MODEL ,NAME_DIR_BEST_MODEL + '.pt')
# # Path to best performing model so far
# CONFIG_BEST_MODEL = path.join(TRAINED_MODELS, NAME_DIR_BEST_MODEL, "config.txt")

# Path to project directory
PROJECT_DIR_PATH = path.dirname(path.dirname(path.abspath(__file__)))


#New Data to ne annptated
NEW_CSV_FILE = path.join(PROJECT_DIR_PATH, "data", "new.csv")

# Train and test csv files
TRAIN_CSV_FILE = path.join(PROJECT_DIR_PATH, "data", "traindata.DataTrain1766.csv")
TEST_CSV_FILE= path.join(PROJECT_DIR_PATH, "data", "testdata.DataTest1766.csv")

# Processed train/test data
TRAIN_ALL_PROCESSED = path.join(PROJECT_DIR_PATH, "data", "train_all_processed.pickle")
TEST_ALL_PROCESSED = path.join(PROJECT_DIR_PATH, "data", "test_all_processed.pickle")

# Path to trained models directory
TRAINED_MODELS = path.join(PROJECT_DIR_PATH, "trained_models")

# Name of directory of best performing model so far (stored in trained models folder)
NAME_DIR_BEST_MODEL = "bert_sentiment140_1152records_0.86acc_12_44_02"
# Path to best performing model so far
BEST_MODEL = path.join(TRAINED_MODELS ,NAME_DIR_BEST_MODEL ,NAME_DIR_BEST_MODEL + '.pt')
# Path to best performing model so far
CONFIG_BEST_MODEL = path.join(TRAINED_MODELS, NAME_DIR_BEST_MODEL, "config.txt")