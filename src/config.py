"""
Global Configuration File
"""

###############################################################
# Random Seed
###############################################################

RANDOM_STATE = 42


###############################################################
# Train Test Split
###############################################################

TEST_SIZE = 0.20


###############################################################
# Optuna
###############################################################

N_TRIALS = 20


###############################################################
# SHAP
###############################################################

SHAP_SAMPLE_SIZE = 2000


###############################################################
# Threshold Optimization
###############################################################

THRESHOLD_METRIC = "f1"

THRESHOLD_START = 0.10

THRESHOLD_STOP = 0.95

THRESHOLD_STEP = 0.05


###############################################################
# Model
###############################################################

BASELINE_PARAMS = {

    "n_estimators":100,

    "max_depth":6,

    "learning_rate":0.1,

    "subsample":0.8,

    "colsample_bytree":0.8,

    "tree_method":"hist",

    "eval_metric":"logloss",

    "n_jobs":-1

}


###############################################################
# Paths
###############################################################

MODEL_DIR = "models"

OUTPUT_DIR = "outputs"

RAW_DATA_PATH_TRANSACTION = "data/raw/train_transaction.csv"

RAW_DATA_PATH_IDENTITY= "data/raw/train_identity.csv"

PROCESSED_DATA_PATH = "data/processed/processed_train.parquet"


###############################################################
# SHAP Plots
###############################################################

SHAP_SUMMARY_PLOT = "outputs/shap_summary.png"

SHAP_BAR_PLOT = "outputs/shap_bar.png"

SHAP_WATERFALL_PLOT = "outputs/shap_waterfall.png"

SHAP_IMPORTANCE_CSV = "outputs/shap_feature_importance.csv"


###############################################################
# Feature Importance
###############################################################

FEATURE_IMPORTANCE_CSV = "outputs/feature_importance.csv"

FEATURE_IMPORTANCE_PLOT = "outputs/feature_importance.png"
