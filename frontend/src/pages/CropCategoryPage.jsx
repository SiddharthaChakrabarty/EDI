import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import CropDetailCard from '../components/CropDetailCard';
import { useTranslation } from 'react-i18next';
import Header from '../components/Header';
import axios from 'axios';

const CropCategoryPage = () => {
  const { category } = useParams(); // keep for routing, even if unused
  const [recommendations, setRecommendations] = useState([]);
  const [topExplanation, setTopExplanation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { t } = useTranslation();

  useEffect(() => {
    const fetchRecommendationsML = async () => {
      setLoading(true);
      try {
        const lat = parseFloat(localStorage.getItem('latitude'));
        const lon = parseFloat(localStorage.getItem('longitude'));

        // Optional soil (if user stored it)
        const ph = localStorage.getItem("ph");
        const N  = localStorage.getItem("N");
        const P  = localStorage.getItem("P");
        const K  = localStorage.getItem("K");

        const payload = {
          lat, lon,
          top_k: 3,
          locale: localStorage.getItem('languagePreference') || "en-IN"
        };

        // attach soil only if present
        if (ph && N && P && K) {
          payload.ph = parseFloat(ph);
          payload.N  = parseFloat(N);
          payload.P  = parseFloat(P);
          payload.K  = parseFloat(K);
        }

        const response = await axios.post(
          'http://127.0.0.1:5000/recommend',
          payload
        );

        setRecommendations(response.data.top_recommendations || []);
        setTopExplanation(response.data.top_explanation || null);
        setError(null);
      } catch (err) {
        console.error('Error fetching recommendations:', err);
        setError(t('errorFetchingRecommendations'));
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendationsML();
  }, [category]);

  return (
    <>
      <Header name={t('Crop Recommendations')} />

      <div className="p-6">
        {loading && (
          <p className="text-lg text-center text-gray-600">{t('Loading...')}</p>
        )}

        {error && (
          <p className="text-red-500 text-lg text-center mt-4">{t(error)}</p>
        )}

        {!loading && !error && (
          <ul className="flex flex-wrap justify-center gap-6 mt-6">
            {recommendations.map((rec, index) => {
              const isTopCrop =
                topExplanation &&
                topExplanation.crop?.toLowerCase() === rec.crop?.toLowerCase();

              return (
                <CropDetailCard
                  key={index}
                  name={rec.crop}
                  ename={rec.crop}
                  confidence={rec.confidence}
                  explanation={isTopCrop ? topExplanation.explanation : null}
                />
              );
            })}
          </ul>
        )}
      </div>
    </>
  );
};

export default CropCategoryPage;
