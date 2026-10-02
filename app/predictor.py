import os
import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import FraudPreprocessor
from src.model import FraudModel
from src.threshold import ThresholdOptimizer

class FraudPredictor:

    def __init__(self):
        print("Loading Artifacts...")

        self.model= FraudModel()
        self.model.load()

        self.preprocessor= FraudPreprocessor.load()

        self.threshold= ThresholdOptimizer.load()

        print("All artifacts loaded successfully")

    def predict_dataframe(self,df):
        df= df.copy()

        df= engineer_features(df)

        transaction_ids= None

        if "TransactionID" in df.columns:
            transaction_ids= df['TransactionID']

        if "isFraud" in df.columns:
            df= df.drop(columns=['isFraud'])

        df= self.preprocessor.transform(df)

        probabilities= self.model.predict_proba(df)

        predictions= (probabilities>=self.threshold).astype(int)

        result= pd.DataFrame({
            "FraudProbability": probabilities,
            "Prediction": predictions
        })

        if transaction_ids is not None:
            result.insert(0, "TransactionID", transaction_ids.values)

        return result
    
    def predict_csv(self, transaction_path, identity_path):
        transaction= pd.read_csv(transaction_path)
        identity= pd.read_csv(identity_path)

        df= transaction.merge(identity, on= "TransactionID",how="left")

        return self.predict_dataframe(df)


    def save_predictions(self, prediction_df, save_path="outputs/predictions.csv"):
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        prediction_df.to_csv(save_path,index=False)

        print(f"Predictions saved at {save_path}")