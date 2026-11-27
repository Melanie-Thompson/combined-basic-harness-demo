def weather_fetch(city: str) -> str:
    import urllib.request
    import urllib.parse
    import json

    # Geocode city name to lat/lon via Open-Meteo
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
    try:
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo = json.loads(resp.read())
    except Exception as e:
        return f"error: geocoding failed for '{city}': {e}"

    results = geo.get("results")
    if not results:
        return f"error: city '{city}' not found"

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]
    name = results[0].get("name", city)
    country = results[0].get("country", "")

    # Fetch current weather from Open-Meteo (free, no key)
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
    )
    try:
        with urllib.request.urlopen(weather_url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return f"error: weather fetch failed for '{city}': {e}"

    current = data.get("current", {})
    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    code = current.get("weather_code", -1)

    conditions = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "depositing rime fog",
        51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
        61: "slight rain", 63: "moderate rain", 65: "heavy rain",
        71: "slight snow", 73: "moderate snow", 75: "heavy snow",
        80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
    }
    condition = conditions.get(code, f"code {code}")

    location = f"{name}, {country}" if country else name
    return f"Weather in {location}: {temp}°C, {condition}, humidity {humidity}%, wind {wind} km/h"
