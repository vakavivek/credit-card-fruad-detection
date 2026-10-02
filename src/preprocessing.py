import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


class FraudPreprocessor:

    def __init__(self):

        # Columns to drop
        self.columns_to_drop = []

        # Median values
        self.median_imputation = {}

        # Mode values
        self.mode_imputation = {}

        # Label Encoders
        self.label_encoders = {}

        # Frequency Maps
        self.frequency_maps = {}

        # Final feature order
        self.feature_columns = []

        # Label Encoding Columns
        self.label_cols = [
            "ProductCD",
            "card4",
            "card6",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
            "id_12",
            "id_15",
            "id_16",
            "id_28",
            "id_29",
            "id_35",
            "id_36",
            "id_37",
            "id_38",
            "DeviceType",
            "TimeBucket",
            "prev_device"
        ]

        # Frequency Encoding Columns
        self.frequency_cols = [
            "P_emaildomain",
            "R_emaildomain",
            "DeviceInfo",
            "id_31",
            "prev_email"
        ]

    def fit(self, df):
        
        df = df.copy()

        ##################################################
        # Columns to Drop
        ##################################################

        missing_percent = df.isnull().mean() * 100

        high_missing_cols = (
            missing_percent[
                missing_percent > 95
                ]
                .index
                .tolist()
        )

        id_cols = [
            c for c in df.columns
            if c.startswith("id_")
        ]
             
        v_cols = [
            c for c in df.columns
            if c.startswith("V")
        ]

        remaining_missing = (
            df[id_cols + v_cols]
                .isnull()
                .mean()
                * 100
        )

        high_missing_idv = (
            remaining_missing[
                remaining_missing > 85
            ]
            .index
            .tolist()
        )

        self.columns_to_drop = list(
            set(
                high_missing_cols
                + high_missing_idv
                + ["TransactionID", "dist2"]
            )
        )
        

         ##################################################
        # Median Imputation
        ##################################################

        numeric_cols = (
            df.drop(
            columns=self.columns_to_drop,
            errors="ignore"
        )
        .select_dtypes(include=["number"])
        .columns
        )

        for col in numeric_cols:

            self.median_imputation[col] = (
                df[col].median()
            )

        ##################################################
        # Mode Imputation
        ##################################################

        mode_cols = [
            "card4",
            "card6"
        ]

        for col in mode_cols:

            if col in df.columns:

                self.mode_imputation[col] = (
                    df[col].mode()[0]
                )

        
        ##################################################
        # Label Encoders
        ##################################################

        for col in self.label_cols:

            if col not in df.columns:
                continue

            temp = (
                df[col]
                .fillna("Unknown")
                .astype(str)
            )

            le = LabelEncoder()

            le.fit(temp)

            if "Unknown" not in le.classes_:

                le.classes_ = np.append(
                    le.classes_,
                    "Unknown"
                )

            self.label_encoders[col] = le


        ##################################################
        # Frequency Maps
        ##################################################

        for col in self.frequency_cols:

            if col not in df.columns:
                continue

            freq = (
                df[col]
                .fillna("Unknown")
                .astype(str)
                .value_counts(normalize=True)
            )

            self.frequency_maps[col] = freq

        ##################################################
        # Feature Order
        ##################################################

        temp = df.drop(
            columns=self.columns_to_drop,
            errors="ignore"
        )

        self.feature_columns = temp.columns.tolist()


        ###############################################################
    # Label Encoding
    ###############################################################

    def _label_encode(self, df):

        for col, encoder in self.label_encoders.items():

            if col not in df.columns:
                continue

            df[col] = df[col].fillna("Unknown").astype(str)

            df[col] = df[col].where(
                df[col].isin(encoder.classes_),
                "Unknown"
            )

            df[col] = encoder.transform(df[col])

        return df
    
        ###############################################################
    # Frequency Encoding
    ###############################################################

    def _frequency_encode(self, df):

        for col, freq_map in self.frequency_maps.items():

            if col not in df.columns:
                continue

            df[col] = (
                df[col]
                .fillna("Unknown")
                .astype(str)
                .map(freq_map)
                .fillna(0)
            )

        return df


        ###############################################################
    # Align Feature Columns
    ###############################################################

    def _align_columns(self, df):

        for col in self.feature_columns:

            if col not in df.columns:

                df[col] = 0

        df = df[self.feature_columns]

        return df
    
        ###############################################################
    # Transform
    ###############################################################

    def transform(self, df):

        df = df.copy()

        ##################################################
        # Drop Columns
        ##################################################

        df.drop(
            columns=self.columns_to_drop,
            inplace=True,
            errors="ignore"
        )

        ##################################################
        # Unknown Imputation
        ##################################################

        unknown_cols = [
            "DeviceType",
            "DeviceInfo",
            "P_emaildomain",
            "R_emaildomain",
            "prev_email"
        ]

        for col in unknown_cols:

            if col in df.columns:

                df[col] = df[col].fillna("Unknown")

        ##################################################
        # Mode Imputation
        ##################################################

        for col, mode in self.mode_imputation.items():

            if col in df.columns:

                df[col] = df[col].fillna(mode)

        ##################################################
        # Median Imputation
        ##################################################

        for col, median in self.median_imputation.items():

            if col in df.columns:

                df[col] = df[col].fillna(median)

        ##################################################
        # Special Cases
        ##################################################

        if "Prev_transaction_amt" in df.columns:
            df["Prev_transaction_amt"] = (
                df["Prev_transaction_amt"]
                .fillna(0)
            )

        if "time_since_prev_txn" in df.columns:
            df["time_since_prev_txn"] = (
                df["time_since_prev_txn"]
                .fillna(0)
            )

        if "dist1" in df.columns:

            df["dist1_missing"] = (
                df["dist1"]
                .isna()
                .astype(int)
            )

            df["dist1"] = df["dist1"].fillna(-1)

        ##################################################
        # Label Encoding
        ##################################################

        df = self._label_encode(df)

        ##################################################
        # Frequency Encoding
        ##################################################

        df = self._frequency_encode(df)

        ##################################################
        # Remaining Numeric
        ##################################################

        numeric_cols = df.select_dtypes(
            include=["number"]
        ).columns

        for col in numeric_cols:

            if df[col].isnull().sum() > 0:

                median = self.median_imputation.get(col, 0)

                df[col] = df[col].fillna(median)

        ##################################################
        # Align Columns
        ##################################################

        df = self._align_columns(df)

        return df
    
        ###############################################################
    # Fit Transform
    ###############################################################

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)

    def save(self, path="models"):
        os.makedirs(path, exist_ok=True)
        preprocessor_path= os.path.join(path, "preprocessor.pkl")
        joblib.dump(self,preprocessor_path)
        print(f"Preprocessor saved successfully at {preprocessor_path}")


    @staticmethod
    def load(path="models/preprocessor.pkl"):
        preprocessor= joblib.load(path)
        print(f"Preprocessor loaded successfully from {path}")
        return preprocessor