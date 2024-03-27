from flask import Flask, render_template
import requests

app = Flask(__name__)

@app.route('/')
def index():
    # Make API request to fetch data
    api_url = "http://localhost:80/predict"
    response = requests.get(api_url)
    
    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        return render_template('index.html', data=data)
    else:
        return "Failed to fetch data from API"

if __name__ == '__main__':
    app.run(debug=True)
