import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import Header from '../components/Header';
import CategoryCard from '../components/CategoryCard';
import { useTranslation } from 'react-i18next';

const PestChatbot = () => {
    const { t, i18n } = useTranslation();
    const [cropName, setCropName] = useState('');
    const [image, setImage] = useState(null);
    const [responseDetails, setResponseDetails] = useState('');
    const [error, setError] = useState(null);
    const [selectedCategory, setSelectedCategory] = useState('');
    const [disease, setDisease] = useState('');

    useEffect(() => {
        const storedLanguage = localStorage.getItem('languagePreference') || 'en';
        i18n.changeLanguage(storedLanguage).catch(err => console.error('Error changing language:', err));
    }, []);

    const categories = [
        t('Pest and Diseases'),
        t('Pesticide Recommendation'),
        t('Irrigation Schedules'),
        t('Crop Rotation Advice')
    ];

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);

        try {
            if (image) {
                const formData = new FormData();
                formData.append('image', image);

                const response = await axios.post('http://127.0.0.1:5000/predict-disease', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                });

                setResponseDetails(response.data.prediction || '');
                setDisease(response.data.prediction || '');
            } 
        } catch (err) {
            console.error('Error fetching response:', err);
            setError(t('Something went wrong. Please try again.'));
        }
    };

    const handleCategoryClick = async (category) => {
        setSelectedCategory(category);
        setError(null);

        try {
            const response = await axios.post('http://127.0.0.1:5000/crop_steps', {
                crop_name: cropName ? cropName : disease,
                language: localStorage.getItem('languagePreference') || 'en',
                category: category
            });

            setResponseDetails(response.data || '');
        } catch (err) {
            console.error('Error fetching data:', err);
            setError(t('Something went wrong. Please try again.'));
        }
    };

    /** 🔥 Drag and Drop File Handling */
    const onDrop = useCallback((acceptedFiles) => {
        setImage(acceptedFiles[0]);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: 'image/*',
        multiple: false
    });

    return (
        <>
            <Header name={t('Pest Disease Chatbot')} />
            <div className="flex flex-col items-center gap-8 p-6 min-h-screen">
                {/* Form */}
                <form 
                    onSubmit={handleSubmit} 
                    className="bg-white shadow-lg rounded-lg p-6 w-full max-w-lg space-y-4"
                >
                    <div>
                        <label htmlFor="cropName" className="block text-lg font-semibold text-gray-700">
                            {t('Crop Name')}:
                        </label>
                        <input
                            type="text"
                            id="cropName"
                            value={cropName}
                            onChange={(e) => setCropName(e.target.value)}
                            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:outline-none"
                        />
                    </div>

                    {/* 🔥 Drag and Drop File Upload */}
                    <div 
                        {...getRootProps()} 
                        className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer bg-gray-50 hover:bg-gray-100 transition"
                    >
                        <input {...getInputProps()} />
                        {isDragActive ? (
                            <p className="text-green-600 font-semibold">{t('Drop the image here...')}</p>
                        ) : (
                            <p className="text-gray-700">{t('Drag & drop an image here, or click to select a file')}</p>
                        )}
                    </div>

                    {/* Show Selected File Name */}
                    {image && (
                        <p className="text-sm text-gray-600 mt-2 text-center">
                            {t('Selected File')}: <span className="font-semibold">{image.name}</span>
                        </p>
                    )}

                    <button 
                        type="submit" 
                        className="w-full bg-green-600 text-white font-bold py-3 rounded-lg hover:bg-green-700 transition duration-300"
                    >
                        {t('Submit')}
                    </button>
                </form>

                {/* Category Cards */}
                <div className="grid grid-cols-1 gap-6 w-full max-w-4xl">
                    {categories.map((category) => (
                        <CategoryCard
                            key={category}
                            category={category}
                            details={selectedCategory === category ? responseDetails : ''}
                            onClick={() => handleCategoryClick(category)}
                        />
                    ))}
                </div>

                {/* Error Message */}
                {error && <p className="text-red-600 font-semibold">{error}</p>}
            </div>
        </>
    );
};

export default PestChatbot;
