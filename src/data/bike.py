import requests
import json
import os

url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'

data_dir = r'C:\Users\Uporabnik\Desktop\notebooks\inteligVaje\data\raw'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    for item in data:
        item.pop("last_update", None)
        item.pop("status", None)
        item.pop("bonus", None)
        item.pop("banking", None)

        filename = f"{item['name']}.json"

        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                existing_data = json.load(f)

            existing_data.append(item)

            with open(filepath, "w") as f:
                json.dump(existing_data, f, indent=4)
        else:
            with open(filepath, "w") as f:
                json.dump([item], f, indent=4)
else:
    print(f"Failed to fetch data from {url}. Status code: {response.status_code}")