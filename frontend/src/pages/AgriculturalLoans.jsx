import React, { useState } from "react";
import Header from "../components/Header";
import { useTranslation } from "react-i18next";

const loans = [
    {
        name: "SBI Crop Loan",
        description: "Offers financial support for crop production expenses, post-harvest costs, and more.",
        link: "https://sbi.co.in/web/agri-rural/agriculture-banking/crop-loan"
    },
    {
        name: "PNB Kisan Credit Card (KCC)",
        description: "Provides timely credit to farmers for cultivation and other needs.",
        link: "https://www.pnbindia.in/agriculture-credit-schemes.html"
    },
    {
        name: "Bank of Baroda Agriculture Loans",
        description: "Offers various agricultural loans with competitive interest rates.",
        link: "https://www.bankofbaroda.in/business-banking/rural-and-agri/loans-and-advances"
    },
    {
        name: "Axis Bank Kisan Credit Card",
        description: "Committed to offering timely credit to farmers, covering cultivation, farm maintenance, and investment needs.",
        link: "https://www.axisbank.com/agri-and-rural/loans"
    },
    {
        name: "Union Bank of India Short-Term Agriculture Loan",
        description: "Provides quick approvals, flexible terms, and competitive interest rates for short-term agricultural needs.",
        link: "https://www.unionbankofindia.co.in/en/subcatlist/agriculture-loan"
    }
];

const AgriculturalLoans = () => {
    const { t } = useTranslation();
    const [loanAmount, setLoanAmount] = useState(100000);
    const [interestRate, setInterestRate] = useState(7.5);
    const [loanTenure, setLoanTenure] = useState(5);
    const [emi, setEmi] = useState(null);

    const calculateEMI = () => {
        const monthlyRate = interestRate / 12 / 100;
        const months = loanTenure * 12;
        const emiCalc = (loanAmount * monthlyRate * Math.pow(1 + monthlyRate, months)) / (Math.pow(1 + monthlyRate, months) - 1);
        setEmi(emiCalc.toFixed(2));
    };

    return (
        <>
            <Header name={t("Agricultural Loans")} />
            <div className="flex flex-col items-center gap-8 p-6 min-h-screen">
                <div className="w-full max-w-4xl bg-white shadow-lg rounded-lg p-6">
                    <h2 className="text-2xl font-semibold text-green-700">{t("Available Loan Options")}</h2>
                    <div className="grid grid-cols-1 gap-6 mt-4">
                        {loans.map((loan, index) => (
                            <div key={index} className="p-4 border rounded-lg shadow-md">
                                <h3 className="text-lg font-bold text-green-800">{t(loan.name)}</h3>
                                <p className="text-gray-700">{t(loan.description)}</p>
                                <a href={loan.link} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline mt-2 inline-block">
                                    {t("Learn More")}
                                </a>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="w-full max-w-4xl bg-white shadow-lg rounded-lg p-6 mt-6">
                    <h2 className="text-2xl font-semibold text-green-700">{t("EMI Calculator")}</h2>
                    <div className="space-y-4 mt-4">
                        <div>
                            <label className="block text-gray-700">{t("Loan Amount (₹)")}</label>
                            <input type="number" value={loanAmount} onChange={(e) => setLoanAmount(e.target.value)} className="w-full p-2 border rounded" />
                        </div>
                        <div>
                            <label className="block text-gray-700">{t("Interest Rate (% per annum)")}</label>
                            <input type="number" step="0.1" value={interestRate} onChange={(e) => setInterestRate(e.target.value)} className="w-full p-2 border rounded" />
                        </div>
                        <div>
                            <label className="block text-gray-700">{t("Loan Tenure (Years)")}</label>
                            <input type="number" value={loanTenure} onChange={(e) => setLoanTenure(e.target.value)} className="w-full p-2 border rounded" />
                        </div>
                        <button onClick={calculateEMI} className="w-full bg-green-600 text-white font-bold py-2 rounded-lg hover:bg-green-700 transition duration-300">
                            {t("Calculate EMI")}
                        </button>
                        {emi && <p className="text-lg font-semibold text-green-700 mt-2">{t("Monthly EMI")}: ₹{emi}</p>}
                    </div>
                </div>
            </div>
        </>
    );
};

export default AgriculturalLoans;
