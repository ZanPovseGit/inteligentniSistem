import requests
import json
import os


url = 'https://api.open-meteo.com/v1/forecast?latitude=46.5547&longitude=15.6467&current=temperature_2m,rain,weather_code&timezone=Europe%2FBerlin&forecast_days=1'

data_dir = r'C:\Users\Uporabnik\Desktop\notebooks\inteligVaje\data\raw'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

response = requests.get(url)

if response.status_code == 200:

    data = response.json()

    desired_data = {
        "time": data["current"]["time"],
        "temperature_2m": data["current"]["temperature_2m"],
        "rain": data["current"]["rain"],
        "weather_code": data["current"]["weather_code"]
    }

    filename = "weather_data.json"

    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_data = json.load(f)

        existing_data.append(desired_data)

        with open(filepath, "w") as f:
            json.dump(existing_data, f, indent=4)
    else:
        with open(filepath, "w") as f:
            json.dump([desired_data], f, indent=4)
else:
    print(f"Failed to fetch data from {url}. Status code: {response.status_code}")