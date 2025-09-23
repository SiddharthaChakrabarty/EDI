import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const CropDetailCard = ({ name, description, image, ename }) => {
    const { t } = useTranslation();
    const navigate = useNavigate();

    const handleClick = () => {
        navigate(`/crop-detail/${ename}`);
    };

    return (
        <div 
            className="relative bg-white border border-gray-200 rounded-xl shadow-lg p-6 m-4 max-w-[320px] text-center 
                        transition-transform duration-300 ease-in-out transform hover:-translate-y-2 hover:shadow-2xl cursor-pointer"
            onClick={handleClick} // Call handleClick when the card is clicked
        >

            {/* Image Section */}
            {image && (
                <img 
                    src={image} 
                    alt={t(name)} 
                    className="w-full h-[200px] object-cover rounded-lg mb-4 shadow-md transition-transform duration-300 hover:scale-105"
                />
            )}

            {/* Title */}
            <h3 className="text-xl font-bold text-gray-800 mb-2">{t(name)}</h3>

            {/* Description */}
            <p className="text-gray-600 text-sm leading-relaxed">
                {t(description)}
            </p>

        </div>
    );
};

export default CropDetailCard;
