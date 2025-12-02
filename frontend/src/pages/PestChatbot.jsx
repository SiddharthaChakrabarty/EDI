// src/pages/PestChatbot.jsx

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
  const [previewUrl, setPreviewUrl] = useState(null);

  const [responseDetails, setResponseDetails] = useState('');
  const [diagnosis, setDiagnosis] = useState(null);

  const [error, setError] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [disease, setDisease] = useState('');

  // --------------------------------------------------
  // Language setup
  // --------------------------------------------------
  useEffect(() => {
    const storedLanguage = localStorage.getItem('languagePreference') || 'en';
    i18n
      .changeLanguage(storedLanguage)
      .catch(err => console.error('Error changing language:', err));
  }, [i18n]);

  // Clean up preview URL
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const categories = [
    t('Pest and Diseases'),
    t('Pesticide Recommendation'),
    t('Irrigation Schedules'),
    t('Crop Rotation Advice'),
  ];

  // --------------------------------------------------
  // Helper: format inline markdown (**bold**)
  // --------------------------------------------------
  const formatInline = (text) => {
    if (!text) return '';
    // Escape < and > for safety, then handle **bold**
    const escaped = text
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    return escaped.replace(
      /\*\*(.*?)\*\*/g,
      '<span class="font-semibold text-gray-900">$1</span>'
    );
  };

  // --------------------------------------------------
  // Helper: render guidance as structured JSX
  // --------------------------------------------------
  const renderGuidance = (text) => {
    if (!text) return null;

    const lines = text.split('\n');
    const blocks = [];
    let bulletBuffer = [];

    const flushBullets = () => {
      if (bulletBuffer.length === 0) return;
      const ulIndex = blocks.length;
      blocks.push(
        <ul
          key={`ul-${ulIndex}`}
          className="list-disc list-inside space-y-1 mb-2 text-sm text-gray-800"
        >
          {bulletBuffer.map((raw, idx) => {
            const line = raw.replace(/^-\s*/, '');
            return (
              <li
                key={idx}
                dangerouslySetInnerHTML={{ __html: formatInline(line) }}
              />
            );
          })}
        </ul>
      );
      bulletBuffer = [];
    };

    lines.forEach((rawLine) => {
      const line = rawLine.trim();

      if (!line) {
        flushBullets();
        return;
      }

      // Bullet lines
      if (line.startsWith('-')) {
        bulletBuffer.push(line);
        return;
      }

      // Normal / heading lines
      flushBullets();
      const isNumberedHeading = /^[0-9]+[).\)]/.test(line) || line.endsWith(':');

      blocks.push(
        <p
          key={`p-${blocks.length}`}
          className={`text-sm text-gray-800 mb-1 ${
            isNumberedHeading ? 'font-semibold mt-3' : ''
          }`}
          dangerouslySetInnerHTML={{ __html: formatInline(line) }}
        />
      );
    });

    flushBullets();
    return blocks;
  };

  // --------------------------------------------------
  // Submit: Disease detection
  // --------------------------------------------------
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResponseDetails('');
    setSelectedCategory('');

    try {
      if (!image) {
        setError(t('Please upload an image before submitting.'));
        return;
      }

      const formData = new FormData();
      formData.append('image', image);

      // crop hint for backend routing
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

      const diag = {
        source: data.source || 'vit',
        crop: cropName || data.crop || '',
        disease: '',
        confidence: null,
        label: '',
        topk: [],
        guidance: '',
      };

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

        if (data.gemma_guidance) {
          details += `\n\n${t('Guidance')}:\n${data.gemma_guidance}`;
        }

        diag.disease = detectedDisease;
        diag.confidence = conf ?? null;
        diag.label = data.classifier_label || '';
        diag.topk = data.topk || [];
        diag.guidance = data.gemma_guidance || '';
      }
      // --------- GEMMA VISION PATH (source: "gemma-only" etc.) ----------
      else {
        detectedDisease = data.disease || '';
        const guidanceText = data.gemma_guidance || data.prediction || '';

        if (detectedDisease) {
          details += `${t('Detected disease')}: ${detectedDisease}\n\n`;
        }
        details += guidanceText || t('No guidance returned from the model.');

        diag.disease = detectedDisease || t('Not explicitly specified');
        diag.confidence = null;
        diag.label = '';
        diag.topk = [];
        diag.guidance = guidanceText;
      }

      setDisease(detectedDisease);
      setDiagnosis(diag);
      setResponseDetails(details);
    } catch (err) {
      console.error('Error fetching response:', err);
      setError(t('Something went wrong. Please try again.'));
    }
  };

  // --------------------------------------------------
  // Category click -> crop_steps
  // --------------------------------------------------
  const handleCategoryClick = async (category) => {
    setSelectedCategory(category);
    setError(null);

    try {
      const { data } = await axios.post('http://127.0.0.1:5000/crop_steps', {
        crop_name: cropName ? cropName : disease,
        language: localStorage.getItem('languagePreference') || 'en',
        category: category,
      });

      setResponseDetails(data || '');
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(t('Something went wrong. Please try again.'));
    }
  };

  // --------------------------------------------------
  // Drag & drop handling
  // --------------------------------------------------
  const onDrop = useCallback(
    (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (!file) return;

      setImage(file);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(file));
    },
    [previewUrl]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: 'image/*',
    multiple: false,
  });

  // --------------------------------------------------
  // UI helpers
  // --------------------------------------------------
  const confidencePct =
    diagnosis && diagnosis.confidence != null
      ? Math.round(diagnosis.confidence * 100)
      : null;

  const guidanceTitle = selectedCategory
    ? `${t('Guidance')}: ${selectedCategory}`
    : t('AI field guidance');

  return (
    <>
      <Header name={t('Pest and Disease Solutions')} />

      <div className="flex flex-col items-center gap-8 p-6 min-h-screen bg-gray-50">
        {/* ---------- Upload & Crop Form ---------- */}
        <form
          onSubmit={handleSubmit}
          className="bg-white shadow-lg rounded-xl p-6 w-full max-w-lg space-y-4"
        >
          <div>
            <label
              htmlFor="cropName"
              className="block text-lg font-semibold text-gray-700 mb-1"
            >
              {t('Crop Name')}:
            </label>
            <input
              type="text"
              id="cropName"
              value={cropName}
              onChange={(e) => setCropName(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:outline-none bg-gray-50"
              placeholder={t('e.g. Corn, Wheat, Rice')}
            />
          </div>

          {/* Drag & Drop */}
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

          {image && (
            <p className="text-sm text-gray-600 mt-1 text-center">
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

        {/* ---------- Diagnosis + Guidance ---------- */}
        {diagnosis && (
          <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Diagnosis card */}
            <div className="bg-white shadow-md rounded-xl p-5 border border-green-100">
              <div className="flex items-center justify-between mb-3">
                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-green-50 text-green-700 border border-green-100">
                  {diagnosis.source === 'vit'
                    ? t('Model: Gemma Vision')
                    : t('Model: CNN-VGG19')}
                </span>
                {confidencePct !== null && (
                  <span className="text-xs text-gray-500">
                    {t('Confidence')}:{" "}
                    <span className="font-semibold">
                      {confidencePct}%
                    </span>
                  </span>
                )}
              </div>

              <h2 className="text-xl font-bold text-gray-800 mb-1">
                {diagnosis.disease || t('Disease not identified')}
              </h2>

              {diagnosis.crop && (
                <p className="text-sm text-gray-500 mb-3">
                  {t('Crop')}:{" "}
                  <span className="font-semibold text-gray-700">
                    {diagnosis.crop}
                  </span>
                </p>
              )}

              {confidencePct !== null && (
                <div className="mt-2">
                  <div className="flex justify-between text-[11px] text-gray-500 mb-1">
                    <span>{t('Model confidence')}</span>
                    <span>{confidencePct}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className={`h-2 rounded-full ${
                        confidencePct >= 80
                          ? 'bg-green-500'
                          : confidencePct >= 50
                          ? 'bg-yellow-400'
                          : 'bg-red-400'
                      }`}
                      style={{ width: `${confidencePct}%` }}
                    />
                  </div>
                </div>
              )}

              {diagnosis.topk && diagnosis.topk.length > 1 && (
                <div className="mt-4">
                  <p className="text-xs font-semibold text-gray-700 mb-1">
                    {t('Top predictions')}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {diagnosis.topk.map((p) => {
                      const pPct =
                        p.probability != null
                          ? Math.round(p.probability * 100)
                          : null;
                      const isMain =
                        p.label === diagnosis.label ||
                        p.label === diagnosis.disease;
                      return (
                        <span
                          key={p.label}
                          className={`inline-flex items-center px-2.5 py-1 rounded-full text-[11px] border ${
                            isMain
                              ? 'bg-green-50 border-green-400 text-green-800'
                              : 'bg-gray-50 border-gray-200 text-gray-600'
                          }`}
                        >
                          {p.label}
                          {pPct !== null && (
                            <span className="ml-1 opacity-80">
                              {pPct}%
                            </span>
                          )}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

              {previewUrl && (
                <div className="mt-4">
                  <p className="text-xs font-semibold text-gray-600 mb-1">
                    {t('Uploaded image')}
                  </p>
                  <img
                    src={previewUrl}
                    alt="Uploaded leaf"
                    className="w-full rounded-lg border border-gray-100 object-cover max-h-48"
                  />
                </div>
              )}
            </div>

            {/* Guidance panel */}
            <div className="bg-white shadow-md rounded-xl p-5 border border-gray-100 lg:col-span-2">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-gray-800">
                  {guidanceTitle}
                </h3>
                {selectedCategory && (
                  <span className="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-600">
                    {t('From category')}: {selectedCategory}
                  </span>
                )}
              </div>
              <p className="text-xs text-gray-500 mb-3">
                {t(
                  'This guidance is generated by the AI model. Please validate with a local agronomist before taking actions.'
                )}
              </p>

              <div className="mt-2 text-sm text-gray-800 leading-relaxed">
                {responseDetails
                  ? renderGuidance(responseDetails)
                  : (
                    <p className="text-gray-500 text-sm">
                      {t('No guidance available yet.')}
                    </p>
                  )}
              </div>
            </div>
          </div>
        )}

        {/* ---------- Category cards ---------- */}
        <div className="grid grid-cols-1 gap-6 w-full max-w-4xl mt-4">
          {categories.map((category) => (
            <CategoryCard
              key={category}
              category={category}
              details={selectedCategory === category ? responseDetails : ''}
              onClick={() => handleCategoryClick(category)}
            />
          ))}
        </div>

        {/* ---------- Error ---------- */}
        {error && (
          <p className="text-red-600 font-semibold mt-2">
            {error}
          </p>
        )}
      </div>
    </>
  );
};

export default PestChatbot;
