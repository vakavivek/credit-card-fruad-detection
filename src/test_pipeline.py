import pandas as pd
import numpy as np

from src.config import RAW_DATA_PATH_IDENTITY,RAW_DATA_PATH_TRANSACTION
from src.feature_engineering import engineer_features
from src.preprocessing import FraudPreprocessor
from src.model import FraudModel
from src.threshold import ThresholdOptimizer

def main():
    print("="*60)
    print("TESTING TRAINED PIPELINE")
    print("="*60)

    print("\nLoading Dataset...")
    transaction= pd.read_csv(RAW_DATA_PATH_TRANSACTION)
    identity= pd.read_csv(RAW_DATA_PATH_IDENTITY)

    df= transaction.merge(
        identity,
        on="TransactionID",
        how="left"
    )

    print(f"Dataset Shape: {df.shape}")

    print("Running Feature Engineering...")
    df= engineer_features(df)

    sample= df.iloc[[0]].copy()
    actual_label= sample['isFraud'].values[0]
    X_sample= sample.drop(columns=['isFraud'])

    print("Loading preprocessor")
    preprocessor= FraudPreprocessor.load()
    print("\nLoading Model...")

    model= FraudModel()
    model.load()

    print("\nLoading Threshold..")
    threshold= ThresholdOptimizer.load()

    X_sample= preprocessor.transform(X_sample)

    probabilty= model.predict_proba(X_sample)[0]

    prediction= int(probabilty>=threshold)

    print("\n Prediction Results")
    print("-"*60)

    print(f"Actual label: {actual_label}")
    print(f"Actual Probabilty: {probabilty:.4f}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Prediction: {prediction}")

    if prediction==actual_label:
        print("\n Prediction Successful")

    else:
        print("\n Prediction Completed (Prediction!=actual_label)")


if __name__=="__main__":
    main()