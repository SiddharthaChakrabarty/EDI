import React, { useState } from 'react';
import axios from 'axios'; 
import { useTranslation } from 'react-i18next';
import CategoryCard from '../components/CategoryCard';
import Header from '../components/Header';

const CropDetailPage = () => {
    const { t } = useTranslation();

    const category = [
        t('Site Selection and Preparation'), t('Seed Selection'), t('Planting'),
        t('Water Management'), t('Nutrient Management'), t('Pest and Disease Management'),
        t('Weed Control'), t('Crop Maintenance'), t('Harvesting'), t('Post-Harvest Handling')
    ];

    const [cropName] = useState(window.location.pathname.split('/').pop()); 
    const [responseDetails, setResponseDetails] = useState({});
    const [selectedCategory, setSelectedCategory] = useState('');
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleCategoryClick = async (category) => {
        setSelectedCategory(category);
        setLoading(true);
        setError(null);

        try {
            const response = await axios.post('http://127.0.0.1:5000/crop_steps', {
                crop_name: cropName,
                language: localStorage.getItem('languagePreference') || 'en',
                category: category
            });

            setResponseDetails(prevDetails => ({
                ...prevDetails,
                [category]: response.data
            }));
        } catch (err) {
            console.error('Error fetching data:', err);
            setError(t('Failed to fetch details. Please try again later.'));
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <Header name={t('Growing Steps')} />

            <div className="p-6 max-w-4xl mx-auto">
                {loading && <p className="text-lg text-gray-700">{t('Loading...')}</p>}
                {error && <p className="text-lg text-red-600 font-bold">{t(error)}</p>}

                <div className="flex flex-wrap justify-center gap-6">
                    {category.map(cat => (
                        <CategoryCard
                            key={cat}
                            category={cat}
                            details={responseDetails[cat] || (selectedCategory === cat ? t('Loading...') : '')} 
                            onClick={() => handleCategoryClick(cat)} 
                        />
                    ))}
                </div>
            </div>
        </>
    );
};

export default CropDetailPage;
