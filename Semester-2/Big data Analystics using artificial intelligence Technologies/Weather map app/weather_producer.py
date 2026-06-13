import requests
import json
from kafka import KafkaProducer
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-producer")

# OpenWeather API details
API_KEY = "7b727378e7fa6fd9ac5ffd632195e515"  # Your API key
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Kafka Producer setup
try:
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=5
    )
    logger.info("Connected to Kafka broker")
except Exception as e:
    logger.error(f"Failed to connect to Kafka: {e}")
    exit(1)

def get_weather_data(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API error: {e}")
        return None

while True:
    city = "Chennai"
    weather_data = get_weather_data(city)
    
    if weather_data and weather_data.get("cod") == 200:
        weather_info = {
            "city": city,
            "temperature": weather_data['main']['temp'],
            "humidity": weather_data['main']['humidity'],
            "pressure": weather_data['main']['pressure'],
            "weather": weather_data['weather'][0]['main'],
            "description": weather_data['weather'][0]['description'],
            "timestamp": weather_data['dt']
        }
        
        try:
            producer.send('weather_topic', value=weather_info)
            producer.flush()
            logger.info(f"Sent weather data: {weather_info}")
        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")
    else:
        logger.warning(f"Invalid data received for {city}")
    
    time.sleep(30)  # Wait 30 seconds between requests