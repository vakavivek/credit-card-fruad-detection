# Credit Card Fraud Detection

An end-to-end machine-learning project for batch credit-card fraud prediction. It trains an XGBoost classifier on transaction and identity data, creates behavioural and time-based features, selects a fraud decision threshold, and produces evaluation and SHAP explainability outputs. A FastAPI service supports predictions from uploaded CSV files; Kafka producer and consumer scripts are included for streaming experiments.

## What is included

- **Feature engineering** – amount, time-of-day, historical card activity, velocity, behavioural-change, and z-score features.
- **Preprocessing** – missing-value handling, categorical label encoding, frequency encoding, and feature alignment.
- **Modeling** – class-weighted XGBoost, Optuna hyperparameter tuning, and a chronological train/test split.
- **Threshold optimisation** – chooses a decision threshold by precision, recall, or F1 score (F1 by default).
- **Evaluation and explainability** – precision, recall, F1, ROC-AUC, PR-AUC, feature importance, and SHAP plots.
- **Serving** – FastAPI endpoint for two-file batch predictions plus Kafka producer/consumer scripts.

## Project layout

```text
Credit_card/
├── app/                    # FastAPI app, batch predictor, Kafka scripts
├── data/raw/               # Training and test transaction/identity CSV files
├── models/                 # Saved model, preprocessor, and threshold artifacts
├── notebooks/              # Data understanding, EDA, features, preprocessing, training
├── outputs/                # Predictions, feature importance, and SHAP outputs
├── src/                    # Training and ML pipeline source code
├── docker-compose.yml      # Kafka and ZooKeeper services
└── requirements.txt
```

## Quick start

Run all commands from the project root.

```bash
cd /Users/vivekvaka/Desktop/Credit_card
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart kafka-python optuna joblib
```

The second install command supplies runtime packages used by the API, Kafka scripts, persistence layer, and tuning code that are not currently listed in `requirements.txt`.

Pretrained artifacts are already available in `models/`:

- `fraud_model.pkl`
- `preprocessor.pkl`
- `best_threshold.pkl`

## Train the model

Place the training files at:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

Then run:

```bash
python -m src.train
```

The training pipeline merges the two files on `TransactionID`, sorts by `TransactionDT`, reserves the most recent 20% for testing, tunes XGBoost with 20 Optuna trials, optimises the classification threshold for F1, and saves model artifacts and explanation outputs.

## Run the trained-pipeline smoke test

```bash
python -m src.test_pipeline
```

This loads the saved artifacts and scores one transaction from the training data.

## Run the API

```bash
uvicorn app.main:app --reload
```

The service is then available at `http://127.0.0.1:8000`.

```bash
curl http://127.0.0.1:8000/health
```

To request a batch prediction, upload matching transaction and identity CSV files:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "transaction_file=@data/raw/test_transaction.csv" \
  -F "identity_file=@data/raw/test_identity.csv"
```

The predictor merges inputs on `TransactionID` and writes results to `outputs/predictions.csv`. Each result contains `TransactionID` (when supplied), `FraudProbability`, and `Prediction`.

## Kafka experiment

Start Kafka and ZooKeeper:

```bash
docker compose up -d
```

In separate terminals, run the consumer and then the producer:

```bash
python -m app.consumer
python -m app.producer
```

Kafka predictions are written to `outputs/kafka_predictions.csv`.

## Outputs

| Location | Description |
| --- | --- |
| `models/threshold_results.csv` | Precision, recall, and F1 across evaluated thresholds |
| `outputs/feature_importance.csv` | XGBoost feature-importance ranking |
| `outputs/shap_feature_importance.csv` | Mean absolute SHAP feature importance |
| `outputs/shap_summary.png` | SHAP feature-impact summary |
| `outputs/shap_bar.png` | SHAP global-importance chart |
| `outputs/shap_waterfall.png` | SHAP explanation for one prediction |
| `outputs/predictions.csv` | Latest batch API prediction results |

## Configuration

Edit `src/config.py` to adjust the random seed, test-set share, number of Optuna trials, SHAP sample size, threshold search range, and output paths. The default threshold metric is F1.

## Current implementation notes

- The API response currently counts `prediction_df['prediction']`, while the predictor creates `Prediction` with an uppercase `P`. Rename that reference before relying on the `fraud_transactions` value in the API response.
- The Kafka producer uses topic `fraud_transaction`, whereas the consumer subscribes to `fraud_transactions`. Set both scripts to the same topic before running the stream.
- The project uses relative paths for data, models, temporary uploads, and outputs. Start commands from the repository root.

## Disclaimer

This project is intended for experimentation and education. Fraud scores should not be the only basis for production financial decisions without data governance, monitoring, privacy review, fairness assessment, and human oversight.
