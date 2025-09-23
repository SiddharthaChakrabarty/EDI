import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CropCard from '../components/CropCard';
import { useTranslation } from 'react-i18next';
import Header from '../components/Header';

const cropCategories = [
    { title: 'Vegetables', image: 'images/vegetable.png' },
    { title: 'Fruits', image: 'images/fruits.png' },
    { title: 'Cereal Crops', image: 'images/cereal.png' },
    { title: 'Legumes', image: 'images/legumes.png' },
    { title: 'Oil Crops', image: 'images/oil.png' },
    { title: 'Spices', image: 'images/spices.png' },
    { title: 'Fiber Crops', image: 'images/fiber.png' },
    { title: 'Medicinal Crops', image: 'images/medicine.png' }
];

const CropRecommendation = () => {
    const { t, i18n } = useTranslation();
    const [latitude, setLatitude] = useState('');
    const [longitude, setLongitude] = useState('');
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const language = localStorage.getItem('languagePreference') || 'en';
        i18n.changeLanguage(language).catch(err => setError(t('errorChangingLanguage', { err: err.message })));

        const storedLatitude = localStorage.getItem('latitude');
        const storedLongitude = localStorage.getItem('longitude');
        if (storedLatitude && storedLongitude) {
            setLatitude(parseFloat(storedLatitude));
            setLongitude(parseFloat(storedLongitude));
        }
    }, []);

    const handleCategoryClick = (category) => {
        navigate(`/crop/${category.title}`, {
            state: { latitude, longitude }
        });
    };

    return (
        <>
            <Header name={t('Type of Crop')} />
            <div className="p-5">
                {latitude && longitude ? (
                    <div className="flex flex-wrap justify-center gap-5">
                        {cropCategories.map((category) => (
                            <CropCard
                                key={category.title}
                                title={t(category.title)}
                                image={category.image}
                                onClick={() => handleCategoryClick(category)}
                            />
                        ))}
                    </div>
                ) : (
                    <p className="text-center text-lg text-gray-700">{t('locationNotAvailable')}</p>
                )}
                {error && <p className="text-center text-lg text-red-500 mt-5">{error}</p>}
            </div>
        </>
    );
};

export default CropRecommendation;
