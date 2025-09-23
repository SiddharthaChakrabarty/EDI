import React, { useEffect, useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import axios from "axios";
import { Bar, Line } from "react-chartjs-2";
import "chart.js/auto";
import Header from "../components/Header";

const MarketTrends = () => {
  const { t } = useTranslation();
  const [bestSellers, setBestSellers] = useState([]);
  const [worstSellers, setWorstSellers] = useState([]);
  const [allCrops, setAllCrops] = useState([]);
  const [selectedCrop, setSelectedCrop] = useState("Rice");
  const [forecastData, setForecastData] = useState(null);
  
  const chartContainerRef = useRef(null);
  const [chartWidth, setChartWidth] = useState(800);

  useEffect(() => {
    axios
      .get("http://localhost:5000/best_worst_sellers")
      .then((res) => {
        const { best_sellers, worst_sellers } = res.data;
        setBestSellers(best_sellers);
        setWorstSellers(worst_sellers);

        const bestNames = best_sellers.map((bs) => bs.Crop);
        const worstNames = worst_sellers.map((ws) => ws.Crop);
        const uniqueCrops = Array.from(new Set([...bestNames, ...worstNames]));
        setAllCrops(uniqueCrops);
      })
      .catch((err) => console.error("Best/Worst error:", err));
  }, []);

  useEffect(() => {
    axios
      .get(`http://localhost:5000/forecast?crop=${selectedCrop}&periods=7`)
      .then((res) => setForecastData(res.data))
      .catch((err) => console.error("Forecast error:", err));
  }, [selectedCrop]);

  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    const resizeObserver = new ResizeObserver(() => {
      setChartWidth(chartContainerRef.current.clientWidth);
    });

    resizeObserver.observe(chartContainerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  const handleCropChange = (e) => {
    setSelectedCrop(e.target.value);
  };

  const bestSellersData = {
    labels: bestSellers.map((item) => item.Crop),
    datasets: [
      {
        label: t("Best Sellers (kg)"),
        data: bestSellers.map((item) => item.TotalSales),
        backgroundColor: "rgba(54,162,235,0.6)",
      },
    ],
  };

  const worstSellersData = {
    labels: worstSellers.map((item) => item.Crop),
    datasets: [
      {
        label: t("Worst Sellers (kg)"),
        data: worstSellers.map((item) => item.TotalSales),
        backgroundColor: "rgba(255,99,132,0.6)",
      },
    ],
  };

  let lineLabels = [];
  let historicalValues = [];
  let forecastLabels = [];
  let forecastValues = [];

  if (forecastData) {
    historicalValues = forecastData.historical.map((d) => d.Actual);
    forecastValues = forecastData.forecast.map((d) => d.Forecast);

    const histDates = forecastData.historical.map((d) => d.Date);
    const foreDates = forecastData.forecast.map((d) => d.Date);

    lineLabels = [...histDates, ...foreDates];
    forecastLabels = foreDates;
  }

  const lineChartData = {
    labels: lineLabels,
    datasets: [
      {
        label: t("Historical Sales (kg)"),
        data: [
          ...historicalValues,
          ...Array(forecastLabels.length).fill(null),
        ],
        borderColor: "blue",
        backgroundColor: "rgba(0,0,255,0.2)",
        tension: 0.1,
      },
      {
        label: t("Forecast (kg)"),
        data: [
          ...Array(historicalValues.length).fill(null),
          ...forecastValues,
        ],
        borderColor: "orange",
        backgroundColor: "rgba(255,165,0,0.2)",
        tension: 0.1,
      },
    ],
  };

  return (
    <>
      <Header name={t("Time Series Based Market Forecasting")} />
      <div className="min-h-screen p-6">
        <div className="max-w-6xl mx-auto">
          {/* Best & Worst Sellers */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white shadow-lg p-6 rounded-xl">
              <h2 className="text-xl font-semibold text-center text-blue-600 mb-4">
                {t("Top 3 Best-sellers")}
              </h2>
              <Bar data={bestSellersData} />
            </div>
            <div className="bg-white shadow-lg p-6 rounded-xl">
              <h2 className="text-xl font-semibold text-center text-red-600 mb-4">
                {t("Top 3 Worst-sellers")}
              </h2>
              <Bar data={worstSellersData} />
            </div>
          </div>

          {/* Forecast Section */}
          <div className="mt-12 bg-white shadow-lg p-8 rounded-xl text-center">
            <h2 className="text-2xl font-bold text-gray-700">
              {t("Future Trend Forecast")}
            </h2>
            <p className="text-gray-500">{t("Select a crop to see predictions")}</p>

            {/* Dropdown for crop selection */}
            <div className="mt-4">
              <select
                value={selectedCrop}
                onChange={handleCropChange}
                className="px-4 py-2 border border-gray-400 rounded-lg"
              >
                {allCrops.map((crop) => (
                  <option key={crop} value={crop}>
                    {crop}
                  </option>
                ))}
              </select>
            </div>

            {forecastData && (
              <div
                ref={chartContainerRef}
                className="mt-6 w-full max-w-4xl mx-auto h-auto"
                style={{ minHeight: "300px" }}
              >
                <Line
                  data={lineChartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                  }}
                  width={chartWidth}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default MarketTrends;
