import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next'; // Import for i18n

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { t } = useTranslation(); // Initialize translation function

  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Sidebar */}
      <div
        className={`fixed top-0 left-0 h-full w-64 bg-[#6B8E23] text-white transform ${isOpen ? 'translate-x-0' : '-translate-x-full'
          } transition-transform duration-300 ease-in-out z-50 p-6 overflow-y-auto shadow-lg`}
      >
        <button
          className="text-white text-2xl font-bold absolute top-4 right-4 focus:outline-none"
          onClick={toggleSidebar}
        >
          ×
        </button>
        <nav className="mt-8">
          <ul className="space-y-4">
            <li>
              <Link to="/" className="text-white text-lg hover:underline">{t('Home')}</Link>
            </li>
            <li>
              <Link to="/recommendations" className="text-white text-lg hover:underline">{t('agro_climatic_recommendation')}</Link>
            </li>
            <li>
              <Link to="/pest-chatbot" className="text-white text-lg hover:underline">{t('cnn_plant_disease')}</Link>
            </li>
            <li>
              <Link to="/chatbot" className="text-white text-lg hover:underline">{t('KrishiSahayak')}</Link>
            </li>
            <li>
              <Link to="/weather" className="text-white text-lg hover:underline">{t('Weather Forecast')}</Link>
            </li>
            <li>
              <Link to="/crop-calendar" className="text-white text-lg hover:underline">{t('Crop Calendar')}</Link>
            </li>
            <li>
              <Link to="/market-trends" className="text-white text-lg hover:underline">{t('Time Series Based Market Forecasting')}</Link>
            </li>
            <li>
              <Link to="/yield-pred" className="text-white text-lg hover:underline">{t('SVM Enhanced Precision Yield Forecasting')}</Link>
            </li>
            <li>
              <Link to="/agri-loans" className="text-white text-lg hover:underline">{t('Agricultural Loans')}</Link>
            </li>
            <li>
              <Link to="/complaints" className="text-white text-lg hover:underline">{t('crowdsourced_farm_reporting')}</Link>
            </li>
            <li>
              <Link to="/investments" className="text-white text-lg hover:underline">{t('Micro Investments')}</Link>
            </li>
          </ul>
        </nav>
      </div>

      {/* Hamburger Menu Button */}
      <button
        className="fixed top-4 left-4 text-black text-3xl font-bold bg-transparent border-none cursor-pointer focus:outline-none z-50"
        onClick={toggleSidebar}
      >
        {!isOpen ? '☰' : ''}
      </button>
    </>
  );
};

export default Sidebar;
