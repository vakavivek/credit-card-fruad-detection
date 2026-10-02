import pandas as pd

from model import FraudModel
from preprocessing import FraudPreprocessor
from threshold import ThresholdOptimizer
from feature_engineering import engineer_features


class FraudPredictor:

    def __init__(self):

        print("Loading Model...")

        self.model = FraudModel()

        self.model.load()

        print("Loading Preprocessor...")

        self.preprocessor = FraudPreprocessor.load()

        print("Loading Threshold...")

        self.threshold = ThresholdOptimizer.load()

        print("Prediction Pipeline Ready!")

    ###############################################################
    # Predict Single Transaction
    ###############################################################

    def predict(self, transaction):

        """
        transaction : dict

        Returns
        -------

        prediction

        fraud_probability
        """

        if isinstance(transaction, dict):

            df = pd.DataFrame([transaction])

        elif isinstance(transaction, pd.DataFrame):

            df = transaction.copy()

        else:

            raise ValueError(
                "Input must be dictionary or DataFrame"
            )

        ##########################################################
        # Feature Engineering
        ##########################################################

        df = engineer_features(df)

        ##########################################################
        # Preprocessing
        ##########################################################

        df = self.preprocessor.transform(df)

        ##########################################################
        # Probability
        ##########################################################

        probability = self.model.predict_proba(df)[0]

        ##########################################################
        # Threshold
        ##########################################################

        prediction = int(
            probability >= self.threshold
        )

        ##########################################################
        # Return
        ##########################################################

        return {

            "fraud_probability": round(
                float(probability),
                4
            ),

            "prediction": prediction

        }

    ###############################################################
    # Batch Prediction
    ###############################################################

    def batch_predict(

        self,

        df

    ):

        df = engineer_features(df)

        df = self.preprocessor.transform(df)

        probabilities = self.model.predict_proba(df)

        predictions = (

            probabilities >= self.threshold

        ).astype(int)

        result = df.copy()

        result["FraudProbability"] = probabilities

        result["Prediction"] = predictions

        return result