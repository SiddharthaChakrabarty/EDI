import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import Header from "../components/Header";

const CropCalendarPage = () => {
  const { t } = useTranslation();
  const [cropName, setCropName] = useState("");
  const [cropCalendar, setCropCalendar] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);


  const fetchCropCalendar = async () => {
    if (!cropName.trim()) {
      setError(t("Enter a crop name"));
      return;
    }

    setLoading(true);
    setError(null);

    // Retrieve latitude & longitude from localStorage
    const latitude = localStorage.getItem("latitude");
    const longitude = localStorage.getItem("longitude");

    if (!latitude || !longitude) {
      setError(t("Location data is missing"));
      setLoading(false);
      return;
    }

    const lang = localStorage.getItem("languagePreference") || "en";

    try {
      const response = await fetch("http://localhost:5000/crop-calendar", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ cropName, latitude, longitude, lang }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch crop calendar");
      }

      const result = await response.json();
      setCropCalendar(result.cropCalendar);

      // Store fetched data in localStorage for future use
      localStorage.setItem("selectedCrop", cropName);
      localStorage.setItem("cropCalendar", JSON.stringify(result.cropCalendar));
    } catch (err) {
      setError(t("Error fetching crop calendar") + ": " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Header name={t("Crop Calendar")} />
      <div className="min-h-screen flex justify-center items-center bg-gray-100 px-4">
        <div className="max-w-4xl mx-auto m-8 bg-green-100 p-8 rounded-2xl shadow-2xl text-gray-800">

          {/* Centered Form */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-6">
            <input
              type="text"
              placeholder={t("Enter crop name")}
              value={cropName}
              onChange={(e) => setCropName(e.target.value)}
              className="w-full sm:w-auto px-4 py-2 border border-green-500 rounded-lg text-gray-800"
            />
            <button
              onClick={fetchCropCalendar}
              className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-6 rounded-lg transition"
            >
              {t("Get Calendar")}
            </button>
          </div>

          {error && <div className="text-red-500 text-center mb-4">{error}</div>}

          {loading ? (
            <div className="text-center text-lg">{t("Fetching crop calendar...")}</div>
          ) : cropCalendar.length > 0 ? (
            <div className="mt-6">
              <h3 className="text-2xl font-semibold text-green-600 text-center mb-4">
                {t("Farming Schedule for")} {cropName}
              </h3>
              <pre className="text-lg text-gray-800 whitespace-pre-wrap p-4">
                {cropCalendar}
              </pre>
            </div>
          ) : (
            <div className="text-center text-lg">{t("Enter a crop name")}</div>
          )}
        </div>
      </div>
    </>
  );
};

export default CropCalendarPage;
