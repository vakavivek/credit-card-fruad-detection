import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


class ThresholdOptimizer:

    def __init__(self):

        self.best_threshold = None
        self.results = None
        self.best_metric = None

    #######################################################################
    # Optimize Threshold
    #######################################################################

    def optimize(
        self,
        model,
        X_test,
        y_test,
        metric="f1",
        start=0.10,
        stop=0.95,
        step=0.05
    ):

        metric = metric.lower()

        if metric not in ["precision", "recall", "f1"]:
            raise ValueError(
                "metric must be one of ['precision','recall','f1']"
            )

        y_prob = model.predict_proba(X_test)[:, 1]

        results = []

        thresholds = np.arange(start, stop, step)

        for threshold in thresholds:

            prediction = (y_prob >= threshold).astype(int)

            precision = precision_score(
                y_test,
                prediction,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                prediction,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                prediction,
                zero_division=0
            )

            results.append({

                "Threshold": threshold,

                "Precision": precision,

                "Recall": recall,

                "F1": f1

            })

        self.results = pd.DataFrame(results)

        if metric == "precision":
            idx = self.results["Precision"].idxmax()

        elif metric == "recall":
            idx = self.results["Recall"].idxmax()

        else:
            idx = self.results["F1"].idxmax()

        self.best_threshold = self.results.loc[
            idx,
            "Threshold"
        ]

        self.best_metric = metric

        return self.best_threshold

    #######################################################################
    # Apply Threshold
    #######################################################################

    def predict(
        self,
        probabilities,
        threshold=None
    ):

        if threshold is None:
            threshold = self.best_threshold

        return (probabilities >= threshold).astype(int)

    #######################################################################
    # Results DataFrame
    #######################################################################

    def get_results(self):

        return self.results.copy()

    #######################################################################
    # Save Threshold
    #######################################################################

    def save(
        self,
        path="models"
    ):

        os.makedirs(path, exist_ok=True)

        joblib.dump(

            self.best_threshold,

            os.path.join(
                path,
                "best_threshold.pkl"
            )

        )

        if self.results is not None:

            self.results.to_csv(

                os.path.join(
                    path,
                    "threshold_results.csv"
                ),

                index=False

            )

        print("Threshold artifacts saved successfully.")

    #######################################################################
    # Load Threshold
    #######################################################################

    @staticmethod
    def load(
        path="models/best_threshold.pkl"
    ):

        return joblib.load(path)