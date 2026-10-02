import os
import joblib
import pandas as pd
import numpy as np

from src.config import *

from src.feature_engineering import engineer_features
from src.preprocessing import FraudPreprocessor
from src.model import FraudModel
from src.threshold import ThresholdOptimizer
from src.evaluation import ModelEvaluator
from src.explainabilty import ModelExplainer


###############################################################
# Load Dataset
###############################################################

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

transaction = pd.read_csv(RAW_DATA_PATH_TRANSACTION)

identity= pd.read_csv(RAW_DATA_PATH_IDENTITY)

df= transaction.merge(identity, on="TransactionID",how="left")


print(f"Dataset Shape : {df.shape}")

###############################################################
# Feature Engineering
###############################################################

print("\nRunning Feature Engineering...")

df = engineer_features(df)

###############################################################
# Time Based Train Test Split
###############################################################

print("\nSplitting Dataset...")

df = df.sort_values(
    "TransactionDT"
).reset_index(drop=True)

X = df.drop(columns=["isFraud"])

y = df["isFraud"]

split_index = int(len(df) * (1 - TEST_SIZE))

X_train = X.iloc[:split_index].copy()

X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()

y_test = y.iloc[split_index:].copy()

###############################################################
# Preprocessing
###############################################################

print("\nPreprocessing Data...")

preprocessor = FraudPreprocessor()

preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)

X_test = preprocessor.transform(X_test)

###############################################################
# Baseline Model
###############################################################

print(np.isinf(X_train.values).sum())
inf_cols = X_train.columns[np.isinf(X_train).any()]
for col in inf_cols:

    print(col)

    print(np.isinf(X_train[col]).sum())

print(inf_cols)
print("\nTraining Baseline Model...")

fraud_model = FraudModel()

fraud_model.train_baseline(
    X_train,
    y_train
)

###############################################################
# Validation Split for Optuna
###############################################################

split_index = int(len(X_train) * 0.8)

X_train_optuna = X_train.iloc[:split_index]

X_valid = X_train.iloc[split_index:]

y_train_optuna = y_train.iloc[:split_index]

y_valid = y_train.iloc[split_index:]

###############################################################
# Hyperparameter Tuning
###############################################################

print("\nRunning Optuna...")

best_params = fraud_model.tune_hyperparameters(

    X_train_optuna,

    y_train_optuna,

    X_valid,

    y_valid,

    n_trials=N_TRIALS

)

print(best_params)

###############################################################
# Final Model
###############################################################

print("\nTraining Final Model...")

fraud_model.train_final_model(

    X_train,

    y_train

)

###############################################################
# Threshold Optimization
###############################################################

print("\nOptimizing Threshold...")

threshold_optimizer = ThresholdOptimizer()

best_threshold = threshold_optimizer.optimize(

    fraud_model.model,

    X_test,

    y_test,

    metric=THRESHOLD_METRIC,

    start=THRESHOLD_START,

    stop=THRESHOLD_STOP,

    step=THRESHOLD_STEP

)

print(f"Best Threshold : {best_threshold}")

###############################################################
# Evaluation
###############################################################

print("\nEvaluating Model...")

evaluator = ModelEvaluator()

metrics = evaluator.evaluate(

    fraud_model.model,

    X_test,

    y_test,

    threshold=best_threshold

)

evaluator.print_metrics()

print("\nClassification Report")

print(

    evaluator.get_classification_report(

        y_test

    )

)

print("\nConfusion Matrix")

print(

    evaluator.get_confusion_matrix(

        y_test

    )

)

###############################################################
# Feature Importance
###############################################################

importance = evaluator.feature_importance(

    fraud_model.model,

    X_train

)

evaluator.save_feature_importance(

    fraud_model.model,

    X_train,

    FEATURE_IMPORTANCE_CSV

)

evaluator.plot_feature_importance(

    fraud_model.model,

    X_train

)

###############################################################
# SHAP Explainability
###############################################################

print("\nGenerating SHAP...")

explainer = ModelExplainer()

explainer.explain(

    fraud_model.model,

    X_test,

    sample_size=SHAP_SAMPLE_SIZE

)

explainer.summary_plot(

    SHAP_SUMMARY_PLOT

)

explainer.bar_plot(

    SHAP_BAR_PLOT

)

explainer.waterfall_plot(

    save_path=SHAP_WATERFALL_PLOT

)

explainer.save_feature_importance(

    SHAP_IMPORTANCE_CSV

)

###############################################################
# Save Artifacts
###############################################################

print("\nSaving Artifacts...")

fraud_model.save(MODEL_DIR)

preprocessor.save(MODEL_DIR)

threshold_optimizer.save(MODEL_DIR)

print("\nTraining Pipeline Completed Successfully!")

print("=" * 60)