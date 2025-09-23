import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const HomePage = () => {
  const [language, setLanguage] = useState(localStorage.getItem('languagePreference') || 'en');
  const [latitude, setLatitude] = useState(localStorage.getItem('latitude') || null);
  const [longitude, setLongitude] = useState(localStorage.getItem('longitude') || null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const { i18n, t } = useTranslation();

  useEffect(() => {
    i18n.changeLanguage(language).catch(err => setError(t('errorChangingLanguage', { err: err.message })));
  }, [language]);

  const handleLanguageChange = (event) => {
    setLanguage(event.target.value);
  };

  const handleLanguageSubmit = () => {
    if (language) {
      localStorage.setItem('languagePreference', language);
      i18n.changeLanguage(language).catch(err => setError(t('errorChangingLanguage', { err: err.message })));
    } else {
      setError(t('selectLanguageError'));
    }
  };

  const getLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLatitude(position.coords.latitude);
          setLongitude(position.coords.longitude);
          localStorage.setItem('latitude', position.coords.latitude);
          localStorage.setItem('longitude', position.coords.longitude);
        },
        () => {
          setError(t('locationRetrievalError'));
        }
      );
    } else {
      setError(t('geolocationNotSupported'));
    }
  };

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center text-black text-center p-5 bg-cover bg-center"
      style={{ backgroundImage: "url('/image.png')" }}
    >
      <h1 className="text-4xl font-bold mb-5">{t('Growing Dreams, one seed at a time')}</h1>
      <h2 className="text-2xl mb-6">{t('Select Language')}</h2>

      <div className="flex flex-col items-center gap-4 bg-black/50 p-6 rounded-lg">
        <select
          value={language}
          onChange={handleLanguageChange}
          className="w-52 p-2 rounded border border-gray-300 bg-white text-black"
        >
          <option value="">{t('Select Language')}</option>
          <option value="en">{t('English')}</option>
          <option value="hi">{t('हिंदी')}</option>
          <option value="mr">{t('मराठी')}</option>
          <option value="gu">{t('ગુજરાતી')}</option>
          <option value="bn">{t('বাংলা')}</option>
          <option value="te">{t('తెలుగు')}</option>
          <option value="ta">{t('தமிழ்')}</option>
          <option value="ml">{t('മലയാളം')}</option>
          <option value="kn">{t('ಕನ್ನಡ')}</option>
        </select>
        <button
          onClick={handleLanguageSubmit}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition duration-300"
        >
          {t('Submit')}
        </button>
      </div>

      <button
        onClick={getLocation}
        className="mt-5 px-5 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition duration-300"
      >
        {t('Get My Location')}
      </button>

      {error && <p className="text-red-500 mt-4">{error}</p>}
    </div>
  );
};

export default HomePage;
