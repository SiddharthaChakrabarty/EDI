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
    i18n
      .changeLanguage(storedLanguage)
      .catch(err => console.error('Error changing language:', err));
  }, [i18n]);

  const categories = [
    t('Pest and Diseases'),
    t('Pesticide Recommendation'),
    t('Irrigation Schedules'),
    t('Crop Rotation Advice'),
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResponseDetails('');

    try {
      if (!image) {
        setError(t('Please upload an image before submitting.'));
        return;
      }

      const formData = new FormData();
      formData.append('image', image);

      // Send crop hint so backend can route:
      //   - corn/potato/rice/wheat -> ViT
      //   - others -> Gemma Vision
      if (cropName) {
        formData.append('crop', cropName.toLowerCase());
      }

      const { data } = await axios.post(
        'http://127.0.0.1:5000/predict-disease',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      let detectedDisease = '';
      let details = '';

      // --------- HF ViT PATH (source: "vit") ----------
      if (data.source === 'vit') {
        detectedDisease = data.disease || data.prediction || '';

        const conf = data.classifier_confidence;
        const confText = conf
          ? ` (${(conf * 100).toFixed(2)}% ${t('confidence')})`
          : '';

        if (detectedDisease) {
          details += `${t('Detected disease')}: ${detectedDisease}${confText}\n`;
        }

        if (data.classifier_label) {
          details += `${t('Model label')}: ${data.classifier_label}\n`;
        }

        if (Array.isArray(data.topk) && data.topk.length > 0) {
          details += `\n${t('Top predictions')}:\n`;
          details += data.topk
            .map(
              (p) =>
                `- ${p.label} (${(p.probability * 100).toFixed(1)}%)`
            )
            .join('\n');
        }

        // If backend later adds guidance text even for ViT, we show it
        if (data.gemma_guidance) {
          details += `\n\n${t('Guidance')}:\n${data.gemma_guidance}`;
        }
      }
      // --------- GEMMA VISION PATH (source: "gemma-only" or anything else) ----------
      else {
        detectedDisease = data.disease || '';
        const guidanceText = data.gemma_guidance || data.prediction || '';

        if (detectedDisease) {
          details += `${t('Detected disease')}: ${detectedDisease}\n\n`;
        }

        details += guidanceText || t('No guidance returned from the model.');
      }

      setDisease(detectedDisease);
      setResponseDetails(details);
    } catch (err) {
      console.error('Error fetching response:', err);
      setError(t('Something went wrong. Please try again.'));
    }
  };

  const handleCategoryClick = async (category) => {
    setSelectedCategory(category);
    setError(null);

    try {
      const { data } = await axios.post('http://127.0.0.1:5000/crop_steps', {
        crop_name: cropName ? cropName : disease,
        language: localStorage.getItem('languagePreference') || 'en',
        category: category,
      });

      // crop_steps returns plain text
      setResponseDetails(data || '');
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(t('Something went wrong. Please try again.'));
    }
  };

  // 🔥 Drag and Drop File Handling
  const onDrop = useCallback((acceptedFiles) => {
    setImage(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: 'image/*',
    multiple: false,
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
            <label
              htmlFor="cropName"
              className="block text-lg font-semibold text-gray-700"
            >
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
              <p className="text-green-600 font-semibold">
                {t('Drop the image here...')}
              </p>
            ) : (
              <p className="text-gray-700">
                {t('Drag & drop an image here, or click to select a file')}
              </p>
            )}
          </div>

          {/* Show Selected File Name */}
          {image && (
            <p className="text-sm text-gray-600 mt-2 text-center">
              {t('Selected File')}:{" "}
              <span className="font-semibold">{image.name}</span>
            </p>
          )}

          <button
            type="submit"
            className="w-full bg-green-600 text-white font-bold py-3 rounded-lg hover:bg-green-700 transition duration-300"
          >
            {t('Submit')}
          </button>
        </form>

        {/* Always show latest disease/guidance text */}
        {responseDetails && (
          <div className="w-full max-w-4xl bg-white shadow-md rounded-lg p-4 whitespace-pre-wrap">
            {responseDetails}
          </div>
        )}

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
