import requests
import json
from kafka import KafkaProducer
import time

# OpenWeather API details
API_KEY = "fbf6087b5ea2cf2f8c19a7fc0e39c7bb"
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# Kafka Producer setup
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Function to fetch air pollution data
def get_air_pollution_data(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    response = requests.get(BASE_URL, params=params).json()
    return response

# Infinite loop for continuous data streaming
while True:
    # Coordinates for Chennai
    lat, lon = 13.0827, 80.2707
    air_data = get_air_pollution_data(lat, lon)
    
    if "list" in air_data:
        pollution_info = {
            "city": "Chennai",
            "coordinates": {"lat": lat, "lon": lon},
            "aqi": air_data["list"][0]["main"]["aqi"],  # Air Quality Index
            "components": air_data["list"][0]["components"],  # Pollutant concentrations
            "timestamp": air_data["list"][0]["dt"]
        }
        producer.send('air_topic', value=pollution_info)
        print(f"✅ Air pollution data sent: {pollution_info}")
    else:
        print(f"❌ Failed to fetch air pollution data for Chennai")
    
    time.sleep(30)  # Fetch data every 30 seconds