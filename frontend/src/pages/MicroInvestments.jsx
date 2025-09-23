import React, { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { useTranslation } from "react-i18next";
import Header from "../components/Header";

const investments = [
    { id: 1, name: "PM Kisan Maan Dhan Yojana", progress: 70, invested: 5000, goal: 7000, link: "https://pmkisan.gov.in/" },
    { id: 2, name: "KUSUM Solar Scheme", progress: 50, invested: 3000, goal: 6000, link: "https://mnre.gov.in/solar/schemes/" },
    { id: 3, name: "Dairy Entrepreneurship Development Scheme", progress: 80, invested: 8000, goal: 10000, link: "https://nabard.org/" },
    { id: 4, name: "National Horticulture Mission", progress: 65, invested: 4500, goal: 7000, link: "https://nhm.nic.in/" },
    { id: 5, name: "Poultry Venture Capital Fund", progress: 55, invested: 3500, goal: 6000, link: "https://dahd.nic.in/" }
];

const investmentData = {
    1: [
        { name: "Jan", value: 1200 },
        { name: "Feb", value: 2800 },
        { name: "Mar", value: 2600 },
        { name: "Apr", value: 5000 }
    ],
    2: [
        { name: "Jan", value: 900 },
        { name: "Feb", value: 2100 },
        { name: "Mar", value: 3500 },
        { name: "Apr", value: 3000 }
    ],
    3: [
        { name: "Jan", value: 1500 },
        { name: "Feb", value: 4500 },
        { name: "Mar", value: 6000 },
        { name: "Apr", value: 8000 }
    ],
    4: [
        { name: "Jan", value: 2200 },
        { name: "Feb", value: 1800 },
        { name: "Mar", value: 3800 },
        { name: "Apr", value: 4500 }
    ],
    5: [
        { name: "Jan", value: 800 },
        { name: "Feb", value: 800 },
        { name: "Mar", value: 2500 },
        { name: "Apr", value: 3500 }
    ]
};

const MicroInvestments = () => {
    const [selectedInvestment, setSelectedInvestment] = useState(null);
    const { t } = useTranslation();

    return (
        <>
            <Header name={t("Micro Investments")} />
            <div className="p-6 bg-white to-green-300 min-h-screen text-white">

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {investments.map((investment) => (
                        <div key={investment.id} className="bg-white text-gray-800 p-4 rounded-xl shadow-lg">
                            <h3 className="text-xl font-semibold">{t(investment.name)}</h3>
                            <p className="text-sm text-gray-600">₹{investment.invested} / ₹{investment.goal}</p>
                            <div className="w-full bg-gray-200 h-2 rounded-full">
                                <div className="bg-green-600 h-2 rounded-full" style={{ width: `${investment.progress}%` }}></div>
                            </div>
                            <div className="flex justify-between mt-2">
                                <button
                                    className="bg-green-600 text-white p-2 rounded-lg flex-1 mr-2"
                                    onClick={() => setSelectedInvestment(investment)}
                                >
                                    {t("Invest Now")}
                                </button>
                                <a href={investment.link} target="_blank" rel="noopener noreferrer" className="bg-blue-600 text-white p-2 rounded-lg flex-1 text-center">
                                    {t("Learn More")}
                                </a>
                            </div>
                        </div>
                    ))}
                </div>
                {selectedInvestment && (
                    <div className="mt-8 p-6 bg-white text-gray-800 rounded-lg shadow-lg">
                        <h3 className="text-xl font-semibold">{t(selectedInvestment.name)} {t("Investment")}</h3>
                        <p className="text-gray-600 mb-4">₹{selectedInvestment.invested} / ₹{selectedInvestment.goal}</p>
                        <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={investmentData[selectedInvestment.id]}>
                                <XAxis dataKey="name" stroke="#888" />
                                <YAxis stroke="#888" />
                                <Tooltip />
                                <Line type="monotone" dataKey="value" stroke="#2d6a4f" strokeWidth={3} />
                            </LineChart>
                        </ResponsiveContainer>
                        <div className="mt-4 flex gap-4">
                            <button className="bg-green-600 text-white p-2 rounded-lg flex-1">{t("Add ₹500")}</button>
                            <button className="bg-red-600 text-white p-2 rounded-lg flex-1" onClick={() => setSelectedInvestment(null)}>{t("Close")}</button>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
};

export default MicroInvestments;