import numpy as np
import pandas as pd


def create_amount_features(df):

    df = df.copy()

    df["TransactionAmt_Log"] = np.log1p(df["TransactionAmt"])

    threshold = df["TransactionAmt"].quantile(0.95)

    df["HighValueFlag"] = (
        df["TransactionAmt"] >= threshold
    ).astype(int)

    return df


def create_time_features(df):

    df = df.copy()

    def get_time_bucket(hour):

        if 0 <= hour < 6:
            return "Night"

        elif 6 <= hour < 12:
            return "Morning"

        elif 12 <= hour < 18:
            return "Afternoon"

        return "Evening"

    df["TimeBucket"] = df["Hour"].apply(get_time_bucket)

    df["BusinessHours"] = (
        df["Hour"].between(9, 18)
    ).astype(int)

    return df


def create_previous_transaction_features(df):

    df = df.copy()

    df = (
        df.sort_values(
            ["card1", "TransactionDT"]
        )
        .reset_index(drop=True)
    )

    group = df.groupby("card1")

    df["Prev_transaction_amt"] = (
        group["TransactionAmt"].shift(1)
    )

    df["time_since_prev_txn"] = (
        group["TransactionDT"].diff()
    )

    df["prev_hour"] = (
        group["Hour"].shift(1)
    )

    df["prev_device"] = (
        group["DeviceType"].shift(1)
    )

    df["device_changed"] = (
        df["DeviceType"] != df["prev_device"]
    ).astype(int)

    df["previous_transaction_count"] = (
        group.cumcount()
    )

    return df


def create_running_statistics(df):

    df = df.copy()

    group = df.groupby("card1")

    df["running_avg_amount"] = (
        group["TransactionAmt"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["running_std_amount"] = (
        group["TransactionAmt"]
        .expanding()
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )



    df["running_max_amount"] = (
        group["TransactionAmt"]
        .cummax()
        .shift(1)
    )

    df["running_min_amount"] = (
        group["TransactionAmt"]
        .cummin()
        .shift(1)
    )

    df["amount_vs_running_avg"] = (
        df["TransactionAmt"] /
        df["running_avg_amount"]
    )

    df["amount_vs_running_max"] = (
        df["TransactionAmt"] /
        df["running_max_amount"]
    )

    df["is_first_transaction"] = (
        df["previous_transaction_count"] == 0
    ).astype(int)

    return df


def create_behavior_features(df):

    df = df.copy()

    group = df.groupby("card1")

    df["prev_email"] = (
        group["P_emaildomain"]
        .shift(1)
    )

    df["email_changed"] = (
        df["P_emaildomain"] != df["prev_email"]
    ).astype(int)

    df["prev_addr1"] = (
        group["addr1"]
        .shift(1)
    )

    df["location_changed"] = (
        df["addr1"] != df["prev_addr1"]
    ).astype(int)

    return df


def create_velocity_features(df):

    WINDOW_SECONDS = 300

    result = []

    for card, group in df.groupby("card1"):

        group = group.copy()

        times = group["TransactionDT"].to_numpy()

        amounts = group["TransactionAmt"].to_numpy()

        left = 0

        txn_count = []
        txn_sum = []
        txn_avg = []
        txn_max = []

        for right in range(len(group)):

            while (
                times[right] - times[left]
            ) > WINDOW_SECONDS:

                left += 1

            window_amounts = amounts[left:right]

            txn_count.append(len(window_amounts))

            if len(window_amounts):

                txn_sum.append(window_amounts.sum())

                txn_avg.append(window_amounts.mean())

                txn_max.append(window_amounts.max())

            else:

                txn_sum.append(0)

                txn_avg.append(np.nan)

                txn_max.append(np.nan)

        group["txn_count_last_5min"] = txn_count
        group["amount_last_5min"] = txn_sum
        group["avg_amount_last_5min"] = txn_avg
        group["max_amount_last_5min"] = txn_max

        result.append(group)

    df = (
        pd.concat(result)
        .sort_index()
        .reset_index(drop=True)
    )

    return df


def create_zscore_features(df):

    df = df.copy()

    df["amount_zscore"] = (
        (
            df["TransactionAmt"]
            - df["running_avg_amount"]
        )
        /
        df["running_std_amount"]
    )

    df["amount_zscore"] = (df["amount_zscore"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
)

    return df


def engineer_features(df):

    df['Hour']= (df['TransactionDT']//3600)%24

    df = create_amount_features(df)

    df = create_time_features(df)

    df = create_previous_transaction_features(df)

    df = create_running_statistics(df)

    df = create_behavior_features(df)

    df = create_velocity_features(df)

    df = create_zscore_features(df)

    return df


