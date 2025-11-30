// src/components/CropDetailCard.jsx

import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const CropDetailCard = ({ name, ename, confidence, explanation }) => {
  const { t } = useTranslation();
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(`/crop-detail/${ename}`);
  };

  const confPct =
    confidence !== null && confidence !== undefined
      ? `${Math.round(confidence * 100)}%`
      : '—';

  const hasExplanation = explanation && typeof explanation === 'object';

  return (
    <div
      className="relative bg-white border border-gray-200 rounded-xl shadow-lg p-6 m-4 max-w-[360px] text-left
                 transition-transform duration-300 ease-in-out transform hover:-translate-y-2 hover:shadow-2xl cursor-pointer"
      onClick={handleClick}
    >
      {/* Title */}
      <h3 className="text-xl font-bold text-gray-800 mb-1 text-center">
        {t(name)}
      </h3>

      {/* Confidence */}
      <p className="text-sm text-gray-500 mb-3 text-center">
        {t('Model confidence')}: <span className="font-semibold">{confPct}</span>
      </p>

      {/* Explanation for TOP crop only */}
      {hasExplanation ? (
        <div className="mt-3 bg-green-50 border border-green-200 rounded-lg p-4">
          {/* Summary */}
          {explanation.summary && (
            <p className="text-sm text-gray-800 mb-2">
              <span className="font-semibold">{t('Summary')}: </span>
              {explanation.summary}
            </p>
          )}

          {/* Why suitable now */}
          {explanation.why_suitable_now && (
            <p className="text-sm text-gray-800 mb-2">
              <span className="font-semibold">{t('Why suitable now')}: </span>
              {explanation.why_suitable_now}
            </p>
          )}

          {/* Water & soil */}
          {explanation.water_and_soil && (
            <p className="text-sm text-gray-800 mb-3">
              <span className="font-semibold">{t('Water & soil')}: </span>
              {explanation.water_and_soil}
            </p>
          )}

          {/* Basic management */}
          {Array.isArray(explanation.basic_management) &&
            explanation.basic_management.length > 0 && (
              <div className="mb-2">
                <p className="text-xs font-semibold text-green-900 mb-1">
                  {t('Basic management')}
                </p>
                <ul className="list-disc list-inside text-xs text-gray-800 space-y-1">
                  {explanation.basic_management.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            )}

          {/* Risks */}
          {Array.isArray(explanation.risks) && explanation.risks.length > 0 && (
            <div className="mb-2">
              <p className="text-xs font-semibold text-red-800 mb-1">
                {t('Risks')}
              </p>
              <ul className="list-disc list-inside text-xs text-gray-800 space-y-1">
                {explanation.risks.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Disclaimers */}
          {Array.isArray(explanation.disclaimers) &&
            explanation.disclaimers.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-gray-700 mb-1">
                  {t('Disclaimers')}
                </p>
                <ul className="list-disc list-inside text-[11px] text-gray-700 space-y-1">
                  {explanation.disclaimers.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
        </div>
      ) : (
        <p className="text-gray-600 text-sm mt-2 text-center">
          {t('Tap to view growing steps')}
        </p>
      )}
    </div>
  );
};

export default CropDetailCard;
