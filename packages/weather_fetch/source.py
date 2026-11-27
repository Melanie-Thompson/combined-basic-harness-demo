def weather_fetch(city: str) -> str:
    import urllib.request
    import urllib.parse
    import json
    from difflib import SequenceMatcher

    query = city.strip()
    if not query:
        return "error: no city provided"

    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(query)}&count=10"
    try:
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo = json.loads(resp.read())
    except Exception as e:
        return f"error: geocoding failed for '{city}': {e}"

    results = geo.get("results")
    if not results:
        return f"error: city '{city}' not found"

    # Score each result by how well its name matches the query.
    # The API returns ranked results but the top one may not match
    # (e.g. "Mars" -> "Marseille"). We pick the best name-match
    # among all candidates, letting the API handle the geocoding
    # and us handle the string similarity.
    q_lower = query.lower()
    scored = []
    for r in results:
        name = r.get("name", "").lower()
        ratio = SequenceMatcher(None, q_lower, name).ratio()
        # Boost: query is a prefix of the name or vice versa
        if name.startswith(q_lower) or q_lower.startswith(name):
            ratio = max(ratio, 0.8)
        # Boost: query words all appear in the name
        q_words = q_lower.split()
        if q_words and all(w in name for w in q_words):
            ratio = max(ratio, 0.85)
        scored.append((ratio, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_ratio, best = scored[0]

    if best_ratio < 0.5:
        top_name = results[0].get("name", "?")
        top_country = results[0].get("country", "")
        return f"error: '{city}' is not a known city (closest match: {top_name}, {top_country})"

    lat = best["latitude"]
    lon = best["longitude"]
    name = best.get("name", city)
    country = best.get("country", "")

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

    pop = best.get("population")
    location = f"{name}, {country}" if country else name
    pop_str = f" (population {pop:,})" if pop else " (population unknown)"
    weather = f"Weather in {location}{pop_str}: {temp}°C, {condition}, humidity {humidity}%, wind {wind} km/h"

    if (not pop or pop < 10_000) and query.lower() != name.lower():
        weather += (
            f"\nNote: '{city}' matched a small/obscure place called {name}, {country}."
            f" The user may have meant something else — confirm before presenting this as their answer."
        )
    elif not pop or pop < 10_000:
        weather += (
            f"\nNote: {name} is a very small or obscure place."
            f" If the user likely meant something other than this location, say so."
        )

    return weather
