// src/pages/CropRecommendation.jsx

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';
import Header from '../components/Header';
import CropDetailCard from '../components/CropDetailCard';

const CropRecommendation = () => {
  const { t, i18n } = useTranslation();

  const [form, setForm] = useState({
    latitude: '',
    longitude: '',
    ph: '',
    N: '',
    P: '',
    K: '',
  });

  const [recommendations, setRecommendations] = useState([]);
  const [topExplanation, setTopExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load language + stored location, and auto-fetch if location exists
  useEffect(() => {
    const language = localStorage.getItem('languagePreference') || 'en';
    i18n
      .changeLanguage(language)
      .catch(err =>
        setError(t('errorChangingLanguage', { err: err.message }))
      );

    const storedLatitude = localStorage.getItem('latitude');
    const storedLongitude = localStorage.getItem('longitude');

    if (storedLatitude && storedLongitude) {
      setForm(prev => ({
        ...prev,
        latitude: storedLatitude,
        longitude: storedLongitude,
      }));
      fetchRecommendations(storedLatitude, storedLongitude, {
        ph: '',
        N: '',
        P: '',
        K: '',
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleChange = e => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleUseBrowserLocation = () => {
    if (!navigator.geolocation) {
      setError(t('Geolocation is not supported in this browser.'));
      return;
    }
    setError(null);
    navigator.geolocation.getCurrentPosition(
      pos => {
        const lat = pos.coords.latitude.toFixed(5);
        const lon = pos.coords.longitude.toFixed(5);
        setForm(prev => ({ ...prev, latitude: lat, longitude: lon }));
        localStorage.setItem('latitude', lat);
        localStorage.setItem('longitude', lon);
      },
      err => {
        console.error(err);
        setError(t('Unable to fetch current location.'));
      }
    );
  };

  const fetchRecommendations = async (lat, lon, soilOverrides) => {
    setLoading(true);
    setError(null);

    try {
      const language = localStorage.getItem('languagePreference') || 'en';

      const payload = {
        latitude: lat,
        longitude: lon,
        language,
      };

      // If all soil fields are filled, send them
      const { ph, N, P, K } = soilOverrides;
      if (ph && N && P && K) {
        payload.ph = ph;
        payload.N = N;
        payload.P = P;
        payload.K = K;
      }

      const res = await axios.post(
        'http://127.0.0.1:5000/recommendations',
        payload
      );
      const data = res.data;

      let recs = data.Recommendations || [];

      // Sort by confidence descending (extra safety)
      recs = [...recs].sort(
        (a, b) => (b.confidence || 0) - (a.confidence || 0)
      );

      setRecommendations(recs);
      setTopExplanation(data.top_explanation?.details || null);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(t('errorFetchingRecommendations'));
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = e => {
    e.preventDefault();
    if (!form.latitude || !form.longitude) {
      setError(t('locationNotAvailable'));
      return;
    }

    localStorage.setItem('latitude', form.latitude);
    localStorage.setItem('longitude', form.longitude);

    fetchRecommendations(form.latitude, form.longitude, {
      ph: form.ph,
      N: form.N,
      P: form.P,
      K: form.K,
    });
  };

  // Split into top + remaining for layout
  const topRec = recommendations.length > 0 ? recommendations[0] : null;
  const otherRecs =
    recommendations.length > 1 ? recommendations.slice(1) : [];

  return (
    <>
      <Header name={t('Crop Recommendations')} />

      <div className="p-6 max-w-5xl mx-auto">
        {/* Input form */}
        <form
          onSubmit={handleSubmit}
          className="bg-white rounded-xl shadow-md p-5 mb-6 grid grid-cols-1 md:grid-cols-2 gap-4"
        >
          {/* Location */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t('Latitude')}
            </label>
            <input
              type="number"
              step="any"
              name="latitude"
              value={form.latitude}
              onChange={handleChange}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
              placeholder={t('Enter latitude')}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t('Longitude')}
            </label>
            <input
              type="number"
              step="any"
              name="longitude"
              value={form.longitude}
              onChange={handleChange}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
              placeholder={t('Enter longitude')}
            />
          </div>

          {/* Optional soil fields */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {t('Soil pH (optional)')}
            </label>
            <input
              type="number"
              step="any"
              name="ph"
              value={form.ph}
              onChange={handleChange}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
              placeholder={t('e.g. 6.8')}
            />
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                {t('N (kg/ha)')}
              </label>
              <input
                type="number"
                step="any"
                name="N"
                value={form.N}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-2 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                {t('P (kg/ha)')}
              </label>
              <input
                type="number"
                step="any"
                name="P"
                value={form.P}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-2 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                {t('K (kg/ha)')}
              </label>
              <input
                type="number"
                step="any"
                name="K"
                value={form.K}
                onChange={handleChange}
                className="w-full border border-gray-300 rounded-lg px-2 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="md:col-span-2 flex flex-wrap justify-between items-center gap-3 mt-2">
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleUseBrowserLocation}
                className="px-3 py-2 text-sm rounded-lg border border-green-500 text-green-700 hover:bg-green-50"
              >
                {t('Use current location')}
              </button>
            </div>

            <button
              type="submit"
              className="px-4 py-2 text-sm rounded-lg bg-green-600 text-white font-semibold hover:bg-green-700"
            >
              {t('Get recommendations')}
            </button>
          </div>
        </form>

        {/* Status messages */}
        {loading && (
          <p className="text-center text-gray-600 text-lg">
            {t('Loading...')}
          </p>
        )}

        {error && (
          <p className="text-center text-red-500 text-lg mb-4">{error}</p>
        )}

        {/* Recommendation layout */}
        {!loading && !error && recommendations.length > 0 && (
          <div className="mt-6 space-y-6">
            {/* Top card – primary recommendation */}
            {topRec && (
              <div className="w-full max-w-3xl mx-auto">
                <CropDetailCard
                  isTop
                  name={topRec.name}
                  ename={topRec.ename}
                  confidence={topRec.confidence}
                  explanation={topExplanation}
                />
              </div>
            )}

            {/* Remaining cards below in two-column grid */}
            {otherRecs.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5 max-w-5xl mx-auto">
                {otherRecs.map((rec, index) => (
                  <CropDetailCard
                    key={rec.ename || rec.name || index}
                    name={rec.name}
                    ename={rec.ename}
                    confidence={rec.confidence}
                    explanation={null}
                    isTop={false}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {!loading && !error && recommendations.length === 0 && (
          <p className="text-center text-gray-500 mt-6">
            {t(
              'No recommendations yet. Please enter your location and submit.'
            )}
          </p>
        )}
      </div>
    </>
  );
};

export default CropRecommendation;
