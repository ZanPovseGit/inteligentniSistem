import requests
import pandas as pd
import json

def fetch_json_from_api(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  
        json_data = response.json()
        return json_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def save_json_to_dataframe(json_data):
    if json_data:
        df = pd.DataFrame(json_data)
        return df
    else:
        return None

def save_json_to_file(df, file_path):

    df_filtered = df.drop(['number', 'contract_name'], axis=1)
    

    data_to_save = df_filtered.to_dict(orient='records')
    
    with open(file_path, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=2)
    print(f"JSON file saved successfully at {file_path}")

def main():
    api_url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'
    file_path = r'C:\Users\Uporabnik\Desktop\notebooks\inteligVaje\data\raw\data.json'
    json_data = fetch_json_from_api(api_url)
    if json_data:
        df = save_json_to_dataframe(json_data)
        save_json_to_file(df, file_path)

if __name__ == "__main__":
    main()