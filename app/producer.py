import json
import pandas as pd

from kafka import KafkaProducer

TOPIC_NAME= "fraud_transaction"

producer= KafkaProducer(
    bootstrap_servers= "localhost:9092",
    value_serializer= lambda x:json.dumps(x).encode("utf-8")
)

transaction= pd.read_csv("data/raw/test_transaction.csv")

identity= pd.read_csv("data/raw/test_identity.csv")

df= transaction.merge(identity, on="TransactionID",how="left")

print(f"Sending {len(df)} transactions...")

for _,row in df.iterrows():
    producer.send(TOPIC_NAME, row.to_dict())

producer.flush()

print("All transactions sent successfullly")