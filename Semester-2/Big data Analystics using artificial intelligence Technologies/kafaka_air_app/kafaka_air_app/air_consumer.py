from kafka import KafkaConsumer
import json
import pandas as pd
import os

# Kafka Consumer setup
consumer = KafkaConsumer(
    'air_topic',  # Updated topic name
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='air-group',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# CSV file to store air pollution data
csv_file = "air_pollution_data.csv"

# Check if the CSV file exists; if not, create it with headers
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["city", "latitude", "longitude", "aqi", "components", "timestamp"]).to_csv(csv_file, index=False)

# Process incoming air pollution data
for message in consumer:
    air_data = message.value
    # Convert the air pollution data to a DataFrame
    df = pd.DataFrame([{
        "city": air_data["city"],
        "latitude": air_data["coordinates"]["lat"],
        "longitude": air_data["coordinates"]["lon"],
        "aqi": air_data["aqi"],
        "components": json.dumps(air_data["components"]),  # Store components as a JSON string
        "timestamp": air_data["timestamp"]
    }])
    # Append the data to the CSV file
    df.to_csv(csv_file, mode='a', header=False, index=False)
    print(f"✅ Air pollution data appended to {csv_file}: {air_data}")