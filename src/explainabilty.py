import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class ModelExplainer:

    """
    Handles SHAP Explainability for the Fraud Detection Model.
    """

    def __init__(self):

        self.explainer = None
        self.shap_values = None
        self.X_sample = None

    ##########################################################################
    # Generate SHAP Values
    ##########################################################################

    def explain(
        self,
        model,
        X_test,
        sample_size=2000,
        random_state=42
    ):

        """
        Generate SHAP values.

        Parameters
        ----------
        model : Trained XGBoost Model

        X_test : DataFrame

        sample_size : int

        Returns
        -------
        shap_values
        """

        if len(X_test) > sample_size:

            self.X_sample = X_test.sample(
                sample_size,
                random_state=random_state
            )

        else:

            self.X_sample = X_test.copy()

        self.explainer = shap.Explainer(model)

        self.shap_values = self.explainer(self.X_sample)

        return self.shap_values

    ##########################################################################
    # Summary Plot
    ##########################################################################

    def summary_plot(
        self,
        save_path=None
    ):

        if self.shap_values is None:
            raise ValueError(
                "Run explain() first."
            )

        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            show=False
        )

        if save_path:

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        plt.close()

    ##########################################################################
    # SHAP Bar Plot
    ##########################################################################

    def bar_plot(
        self,
        save_path=None
    ):

        if self.shap_values is None:
            raise ValueError(
                "Run explain() first."
            )

        shap.plots.bar(
            self.shap_values,
            show=False
        )

        if save_path:

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        plt.close()

    ##########################################################################
    # Waterfall Plot
    ##########################################################################

    def waterfall_plot(
        self,
        index=0,
        save_path=None
    ):

        if self.shap_values is None:
            raise ValueError(
                "Run explain() first."
            )

        shap.plots.waterfall(
            self.shap_values[index],
            show=False
        )

        if save_path:

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        plt.close()

    ##########################################################################
    # Force Plot
    ##########################################################################

    def force_plot(
        self,
        index=0
    ):

        if self.shap_values is None:
            raise ValueError(
                "Run explain() first."
            )

        return shap.force_plot(

            self.explainer.expected_value,

            self.shap_values.values[index],

            self.X_sample.iloc[index],

            matplotlib=True

        )

    ##########################################################################
    # Dependence Plot
    ##########################################################################

    def dependence_plot(
        self,
        feature_name,
        interaction_index="auto",
        save_path=None
    ):

        if self.shap_values is None:
            raise ValueError(
                "Run explain() first."
            )

        shap.dependence_plot(

            feature_name,

            self.shap_values.values,

            self.X_sample,

            interaction_index=interaction_index,

            show=False

        )

        if save_path:

            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        plt.close()

    ##########################################################################
    # SHAP Feature Importance
    ##########################################################################

    def feature_importance(self):

        if self.shap_values is None:
            raise ValueError(
                "Run explain() first."
            )

        importance = np.abs(
            self.shap_values.values
        ).mean(axis=0)

        importance_df = pd.DataFrame({

            "Feature": self.X_sample.columns,

            "Importance": importance

        })

        importance_df = importance_df.sort_values(

            by="Importance",

            ascending=False

        )

        return importance_df

    ##########################################################################
    # Save SHAP Feature Importance
    ##########################################################################

    def save_feature_importance(
        self,
        save_path="outputs/shap_feature_importance.csv"
    ):

        importance = self.feature_importance()

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        importance.to_csv(

            save_path,

            index=False

        )

        print(
            f"SHAP Feature Importance saved at {save_path}"
        )