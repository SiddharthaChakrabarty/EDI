import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";

const ComplaintForm = () => {
  const { t } = useTranslation();
  const [text, setText] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setLatitude(pos.coords.latitude.toFixed(4));
          setLongitude(pos.coords.longitude.toFixed(4));
        },
        (err) => {
          console.log("Could not get location:", err);
        }
      );
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    setError("");

    if (!text || !latitude || !longitude) {
      setError(t("error_fill_all_fields"));
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/submit-complaint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          latitude: parseFloat(latitude),
          longitude: parseFloat(longitude),
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setMessage(t("success_complaint_submitted"));
        setText("");
      } else {
        setError(data.error || t("error_submitting_complaint"));
      }
    } catch (err) {
      setError(t("error_network"));
    }
  };

  const handleClustering = async () => {
    setMessage("");
    setError("");
    try {
      const response = await fetch("http://127.0.0.1:5000/run-clustering", {
        method: "POST",
      });
      const data = await response.json();
      if (response.ok) {
        setMessage(t("success_clustering_completed"));
      } else {
        setError(data.error || t("error_running_clustering"));
      }
    } catch (err) {
      setError(t("error_network"));
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white shadow-lg rounded-lg mt-6">
      <h2 className="text-2xl font-bold text-center text-green-700">{t("submit_complaint")}</h2>
      <form onSubmit={handleSubmit} className="space-y-4 mt-4">
        <div>
          <label className="block text-gray-700">{t("complaint_text")}:</label>
          <textarea
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full p-2 border rounded-md focus:ring focus:ring-green-300"
          />
        </div>
        <div>
          <label className="block text-gray-700">{t("latitude")}:</label>
          <input
            type="number"
            step="0.0001"
            value={latitude}
            onChange={(e) => setLatitude(e.target.value)}
            className="w-full p-2 border rounded-md focus:ring focus:ring-green-300"
          />
        </div>
        <div>
          <label className="block text-gray-700">{t("longitude")}:</label>
          <input
            type="number"
            step="0.0001"
            value={longitude}
            onChange={(e) => setLongitude(e.target.value)}
            className="w-full p-2 border rounded-md focus:ring focus:ring-green-300"
          />
        </div>
        <button type="submit" className="w-full bg-green-600 text-white font-bold py-2 rounded-lg hover:bg-green-700 transition duration-300">
          {t("submit_complaint")}
        </button>
      </form>
      <button
        onClick={handleClustering}
        className="w-full bg-blue-600 text-white font-bold py-2 rounded-lg mt-4 hover:bg-blue-700 transition duration-300"
      >
        {t("run_clustering")}
      </button>
      {message && <p className="text-green-700 font-semibold mt-4">{message}</p>}
      {error && <p className="text-red-600 font-semibold mt-4">{error}</p>}
    </div>
  );
};

export default ComplaintForm;
