def weather_fetch(city: str) -> str:
    mock_data = {"London": "14C, overcast", "Tokyo": "22C, sunny", "New York": "18C, partly cloudy"}
    return f"Weather in {city}: {mock_data.get(city, '20C, fair')}"
