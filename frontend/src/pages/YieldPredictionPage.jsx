import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import Header from "../components/Header";

const YieldPredictionPage = () => {
    const { t } = useTranslation();
    const [cropType, setCropType] = useState("Rice");
    const [landSize, setLandSize] = useState("");
    const [fertilizer, setFertilizer] = useState("");
    const [pesticide, setPesticide] = useState("");
    const [predictedYield, setPredictedYield] = useState(null);
    const [errorMsg, setErrorMsg] = useState("");
    const [latitude, setLatitude] = useState(null);
    const [longitude, setLongitude] = useState(null);

    useEffect(() => {
        const storedLatitude = localStorage.getItem("latitude");
        const storedLongitude = localStorage.getItem("longitude");

        if (storedLatitude && storedLongitude) {
            setLatitude(storedLatitude);
            setLongitude(storedLongitude);
        } else if ("geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude.toFixed(4);
                    const lon = position.coords.longitude.toFixed(4);
                    setLatitude(lat);
                    setLongitude(lon);
                    localStorage.setItem("latitude", lat);
                    localStorage.setItem("longitude", lon);
                },
                () => {
                    console.log("Could not get location");
                }
            );
        }
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMsg("");
        setPredictedYield(null);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    cropType,
                    landSize,
                    fertilizer,
                    pesticide,
                    latitude,
                    longitude,
                }),
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "Request failed");
            }

            const data = await response.json();
            setPredictedYield(data.predicted_yield);
        } catch (error) {
            console.error("Error:", error);
            setErrorMsg(error.message);
        }
    };

    return (
        <>
            <Header name={t("SVM Enhanced Precision Yield Forecasting")} />
            <div className="flex flex-col items-center gap-8 p-6 min-h-screen bg-white">
                <form
                    onSubmit={handleSubmit}
                    className="bg-white shadow-lg rounded-lg p-6 w-full max-w-lg space-y-4 border border-green-300"
                >
                    <h2 className="text-2xl font-bold text-green-700 text-center">{t("Predict Your Crop Yield")}</h2>

                    <div>
                        <label className="block text-lg font-semibold text-green-800">{t("Crop Type")}:</label>
                        <select
                            value={cropType}
                            onChange={(e) => setCropType(e.target.value)}
                            className="w-full p-3 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:outline-none"
                        >
                            <option value="Rice">{t("Rice")}</option>
                            <option value="Wheat">{t("Wheat")}</option>
                            <option value="Maize">{t("Maize")}</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-lg font-semibold text-green-800">{t("Land Size (ha)")}:</label>
                        <input
                            type="number"
                            step="0.1"
                            value={landSize}
                            onChange={(e) => setLandSize(e.target.value)}
                            required
                            className="w-full p-3 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:outline-none"
                        />
                    </div>

                    <div>
                        <label className="block text-lg font-semibold text-green-800">{t("Fertilizer (kg/ha)")}:</label>
                        <input
                            type="number"
                            step="0.1"
                            value={fertilizer}
                            onChange={(e) => setFertilizer(e.target.value)}
                            required
                            className="w-full p-3 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:outline-none"
                        />
                    </div>

                    <div>
                        <label className="block text-lg font-semibold text-green-800">{t("Pesticide (kg/ha)")}:</label>
                        <input
                            type="number"
                            step="0.1"
                            value={pesticide}
                            onChange={(e) => setPesticide(e.target.value)}
                            required
                            className="w-full p-3 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:outline-none"
                        />
                    </div>

                    <button
                        type="submit"
                        className="w-full bg-green-600 text-white font-bold py-3 rounded-lg hover:bg-green-700 transition duration-300"
                    >
                        {t("Predict Yield")}
                    </button>
                </form>

                {errorMsg && (
                    <p className="text-red-600 font-semibold">{t("Error")}: {errorMsg}</p>
                )}
                {predictedYield && (
                    <p className="text-lg font-semibold text-green-700">{t("Predicted Yield")}: {predictedYield} kg/ha</p>
                )}
            </div>
        </>
    );
};

export default YieldPredictionPage;