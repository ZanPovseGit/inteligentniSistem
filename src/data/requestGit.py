import os
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
        json_str = json.dumps(json_data, ensure_ascii=False)
        print(json_str.encode('ascii', 'ignore').decode('ascii'))
        df = pd.DataFrame(json_data)
        return df
    else:
        return None

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_json_to_file_per_name(df, df2, folder_path):
    df = df.drop(['number', 'contract_name'], axis=1)
    unique_names = df['name'].unique()
    create_directory_if_not_exists(folder_path)

    for name in unique_names:
        df_filtered = df[df['name'] == name]
        df_filtered = pd.concat([df_filtered]*len(df2), ignore_index=True)
        df_filtered = pd.concat([df_filtered, df2.reset_index(drop=True)], axis=1)
        data_to_save = df_filtered.to_dict(orient='records')

        sanitized_name = ''.join(c if c.isalnum() else '_' for c in name)

        file_path = os.path.join(folder_path, f'{sanitized_name}.json')

        with open(file_path, 'w') as f:
            json.dump(data_to_save, f)



        print(f"JSON data for {name} added to DVC cc")

def main():
    api_url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'
    api_url2 = 'https://api.open-meteo.com/v1/forecast?latitude=46.5547&longitude=15.6467&current=temperature_2m,rain,weather_code&timezone=Europe%2FBerlin&forecast_days=1'
    folder_path = 'data/tempdata/processed'
    json_data = fetch_json_from_api(api_url)
    weather_data = fetch_json_from_api(api_url2)
    os.listdir("data/tempdata/processed/")

    temperature_2m = weather_data['current']['temperature_2m']
    rain = weather_data['current']['rain']
    weather_code = weather_data['current']['weather_code']        
    time = weather_data['current']['time']
        
    df2 = pd.DataFrame({
        'temperature_2m': [temperature_2m],
        'rain': [rain],
        'weather_code': [weather_code],
        'time': [time]
    })


    df = save_json_to_dataframe(json_data)
    save_json_to_file_per_name(df,df2, folder_path)

if __name__ == "__main__":
    main()