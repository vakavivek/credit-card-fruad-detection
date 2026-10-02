import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


class ModelEvaluator:

    def __init__(self):

        self.y_pred = None
        self.y_prob = None
        self.metrics = None

    ###############################################################
    # Evaluate Model
    ###############################################################

    def evaluate(self,model, X_test,y_test, threshold=0.5):

        self.y_prob = model.predict_proba(X_test)[:, 1]

        self.y_pred = (self.y_prob >= threshold).astype(int)

        self.metrics = {

            "Precision": precision_score(
                y_test,
                self.y_pred
            ),

            "Recall": recall_score(
                y_test,
                self.y_pred
            ),

            "F1 Score": f1_score(
                y_test,
                self.y_pred
            ),

            "ROC AUC": roc_auc_score(
                y_test,
                self.y_prob
            ),

            "PR AUC": average_precision_score(
                y_test,
                self.y_prob
            )

        }

        return self.metrics

    ###############################################################
    # Print Metrics
    ###############################################################

    def print_metrics(self):

        print("\nEvaluation Metrics")
        print("-" * 40)

        for key, value in self.metrics.items():

            print(f"{key:<15}: {value:.4f}")

    ###############################################################
    # Classification Report
    ###############################################################

    def get_classification_report(
        self,
        y_test
    ):

        return classification_report(
            y_test,
            self.y_pred
        )

    ###############################################################
    # Confusion Matrix
    ###############################################################

    def get_confusion_matrix(
        self,
        y_test
    ):

        return confusion_matrix(
            y_test,
            self.y_pred
        )

    ###############################################################
    # Feature Importance
    ###############################################################

    def feature_importance(
        self,
        model,
        X_train
    ):

        importance = pd.DataFrame({

            "Feature": X_train.columns,

            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(

            by="Importance",

            ascending=False

        )

        return importance

    ###############################################################
    # Plot Feature Importance
    ###############################################################

    def plot_feature_importance(
        self,
        model,
        X_train,
        top_n=20
    ):

        importance = self.feature_importance(
            model,
            X_train
        )

        importance = importance.head(top_n)

        plt.figure(figsize=(10,8))

        plt.barh(

            importance["Feature"],

            importance["Importance"]

        )

        plt.gca().invert_yaxis()

        plt.xlabel("Importance")

        plt.ylabel("Feature")

        plt.title(f"Top {top_n} Feature Importance")

        plt.tight_layout()

        plt.show()

    ###############################################################
    # Save Feature Importance
    ###############################################################

    def save_feature_importance(
        self,
        model,
        X_train,
        path="outputs/feature_importance.csv"
    ):

        importance = self.feature_importance(
            model,
            X_train
        )
        
        os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

        importance.to_csv(
            path,
            index=False
        )

        print(f"Feature importance saved at {path}")