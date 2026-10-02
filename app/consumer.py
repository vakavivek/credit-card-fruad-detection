import json
import pandas as pd

from kafka import KafkaConsumer
from app.predictor import FraudPredictor

TOPIC_NAME= "fraud_transactions"

consumer= KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers= "localhost:9092",
    auto_offset_reset= "earliest",
    value_deserializer= lambda x: json.loads(
        x.decode("utf-8")
    )
)

predictor= FraudPredictor()

predictions=[]

print("Consumer Started..")

for message in consumer:
    transaction= message.value
    df= pd.DataFrame([transaction])
    result= predictor.predict_dataframe(df)

    predictions.append(result)
    print(
        f"Processed TransactionID:"
        f"{transaction['TransactionID']}"
    )

    if len(predictions)>0:
        final_df= pd.concat(
            predictions,
            ignore_index=True
        )

        predictor.save_predictions(final_df,"outputs/kafka_predictions.csv")


    print("Prediction Completed")