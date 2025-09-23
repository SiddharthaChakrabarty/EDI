import React, { useState, useEffect } from "react";
import axios from "axios";
import { useTranslation } from "react-i18next";
import { Line } from "react-chartjs-2";
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  Title, 
  Tooltip, 
  Legend 
} from "chart.js";
import Header from "../components/Header";

// Register necessary Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const WeatherForecast = () => {
  const { t } = useTranslation();
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [forecast, setForecast] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const savedLat = localStorage.getItem("latitude");
    const savedLon = localStorage.getItem("longitude");

    if (savedLat && savedLon) {
      setLat(savedLat);
      setLon(savedLon);
      fetchWeatherForecast(savedLat, savedLon);
    } else {
      setError(t("Location not found. Please allow location access."));
    }
  }, []);

  const fetchWeatherForecast = async (latitude, longitude) => {
    setLoading(true);
    setError("");
    setForecast(null);

    try {
      const response = await axios.post("http://127.0.0.1:5000/weather-forecast", {
        lat: latitude,
        lon: longitude,
      });
      setForecast(response.data);
    } catch (err) {
      setError(t("Failed to fetch weather forecast. Please try again."));
      console.error("Error fetching weather forecast:", err);
    }

    setLoading(false);
  };

  // Prepare chart data if forecast exists
  const chartData = forecast
    ? {
        labels: forecast.forecast.forecastday.map((day) => day.date),
        datasets: [
          {
            label: t("Max Temperature (°C)"),
            data: forecast.forecast.forecastday.map((day) => day.day.maxtemp_c),
            borderColor: "rgba(255, 0, 0, 0.8)", // Red border
            backgroundColor: "rgba(255, 0, 0, 0.3)", // Light red fill
            fill: true,
            tension: 0.4, // Smooth curves
            pointRadius: 8, // Bigger points
            pointBackgroundColor: "rgba(255, 0, 0, 1)", // Solid red points
            pointBorderColor: "#fff",
            pointHoverRadius: 10, // Larger points on hover
          },
          {
            label: t("Min Temperature (°C)"),
            data: forecast.forecast.forecastday.map((day) => day.day.mintemp_c),
            borderColor: "rgba(0, 0, 255, 0.8)", // Blue border
            backgroundColor: "rgba(0, 0, 255, 0.3)", // Light blue fill
            fill: true,
            tension: 0.4,
            pointRadius: 8,
            pointBackgroundColor: "rgba(0, 0, 255, 1)", // Solid blue points
            pointBorderColor: "#fff",
            pointHoverRadius: 10,
          },
        ],
      }
    : null;

  return (
    <>
      <Header name={t("Weather Forecast")} />
      <div className="min-h-screen w-full p-6 flex flex-col bg-gray-100">

        {/* Error Message */}
        {error && <p className="text-red-600 font-semibold text-center">{error}</p>}

        {/* Loading Indicator */}
        {loading && <p className="text-gray-700 font-semibold text-center">{t("Fetching weather forecast...")}</p>}

        {/* Weather Forecast Display */}
        {forecast && !loading && (
          <div className="w-full px-8">
            {/* Location Header */}
            <h2 className="text-3xl font-bold text-gray-800 text-center mb-6">
              {t("Location")}: {forecast.location.name}, {forecast.location.country}
            </h2>

            {/* Temperature Line Chart */}
            <div className="w-full bg-white p-6 rounded-lg shadow-lg mb-6">
              {chartData && (
                <Line
                  data={chartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: "top",
                        labels: {
                          font: { size: 16, weight: "bold" },
                          color: "#333",
                        },
                      },
                      tooltip: {
                        backgroundColor: "rgba(0, 0, 0, 0.8)",
                        titleFont: { size: 18, weight: "bold" },
                        bodyFont: { size: 16 },
                        padding: 12,
                      },
                    },
                    scales: {
                      x: {
                        grid: { display: false },
                        ticks: { font: { size: 16 }, color: "#000" },
                      },
                      y: {
                        grid: {
                          color: "rgba(200, 200, 200, 0.5)",
                          borderDash: [5, 5],
                        },
                        ticks: { font: { size: 16 }, color: "#000" },
                      },
                    },
                  }}
                  height={400} // Increased height for better visibility
                />
              )}
            </div>

            {/* Weather Cards Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {forecast.forecast.forecastday.map((day, index) => (
                <div
                  key={index}
                  className="bg-gradient-to-r from-green-700 to-yellow-700 text-white p-6 rounded-lg shadow-lg flex flex-col items-center transition transform hover:scale-105"
                >
                  <p className="text-xl font-bold">{day.date}</p>
                  <img src={day.day.condition.icon} alt="Weather Icon" className="w-20 h-20 my-3" />
                  <p className="text-lg font-extrabold">{day.day.condition.text}</p>
                  <p>🌡️ {t("Max Temp")}: <span className="font-extrabold">{day.day.maxtemp_c}°C</span></p>
                  <p>❄️ {t("Min Temp")}: <span className="font-extrabold">{day.day.mintemp_c}°C</span></p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default WeatherForecast;
