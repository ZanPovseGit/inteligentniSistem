// Function to fetch bike data from API
async function fetchBikeData() {
    try {
        const response = await fetch('https://api.example.com/bike-data');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching bike data:', error);
    }
}

// Function to display bike data
async function displayBikeData() {
    const bikeDataContainer = document.getElementById('bike-data');
    const dayContainer = document.getElementById('day');

    // Fetch bike data
    const bikeData = await fetchBikeData();
    if (!bikeData) return;

    // Display bike data
    bikeDataContainer.innerHTML = `
        <p>Temperature: ${bikeData.temperature}°C</p>
        <p>Humidity: ${bikeData.humidity}%</p>
        <p>Wind Speed: ${bikeData.windSpeed} m/s</p>
    `;

    // Display day
    const today = new Date();
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    dayContainer.textContent = days[today.getDay()];
}

// Call function to display bike data
displayBikeData();
