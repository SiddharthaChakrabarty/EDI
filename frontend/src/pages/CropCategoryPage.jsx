import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import CropDetailCard from '../components/CropDetailCard';
import { useTranslation } from 'react-i18next';
import Header from '../components/Header';
import axios from 'axios';

const CropCategoryPage = () => {
    const { category } = useParams();
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const { t } = useTranslation(); 

    useEffect(() => {
        const fetchRecommendations = async () => {
            setLoading(true);
            try {
                const response = await axios.post('http://127.0.0.1:5000/recommendations', {
                    latitude: localStorage.getItem('latitude'),
                    longitude: localStorage.getItem('longitude'),
                    category,
                    language: localStorage.getItem('languagePreference') || 'en'
                });
                setRecommendations(response.data.Recommendations);
                setError(null);
            } catch (err) {
                console.error('Error fetching recommendations:', err);
                setError(t('errorFetchingRecommendations')); 
            } finally {
                setLoading(false);
            }
        };

        fetchRecommendations();
    }, [category]);

    return (
        <>
            <Header name={t('Crop Recommendations')} />
            <div className="p-6">
                {loading && <p className="text-lg text-center text-gray-600">{t('Loading...')}</p>}
                {error && <p className="text-red-500 text-lg text-center mt-4">{t(error)}</p>}
                
                <ul className="flex flex-wrap justify-center gap-6 mt-6">
                    {recommendations.map((rec, index) => (
                        <CropDetailCard
                            key={index}
                            name={t(rec.name)} 
                            ename={rec.ename}
                            description={t(rec.description)} 
                            image={rec.image}
                        />
                    ))}
                </ul>
            </div>
        </>
    );
};

export default CropCategoryPage;
