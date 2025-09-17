import requests

def get_weather(city):
    # Obtener coordenadas
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=es&format=json"
    geo_data = requests.get(geo_url).json()

    if "results" not in geo_data:
        return f"No encontré la ciudad {city}."

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    # Obtener clima actual
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
    weather_data = requests.get(weather_url).json()

    if "current_weather" not in weather_data:
        return "No pude obtener el clima en este momento."

    current = weather_data["current_weather"]
    temp = current["temperature"]
    wind = current["windspeed"]

    return f"El clima en {city} es {temp}°C con viento de {wind} km/h."

print(get_weather("Alginet"))