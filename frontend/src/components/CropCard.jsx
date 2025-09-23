import React from 'react';
import { useTranslation } from 'react-i18next';

const CropCard = ({ title, image, onClick }) => {
    const { t } = useTranslation(); 

    return (
        <div 
            className="flex flex-col items-center cursor-pointer p-2 text-center transition-transform duration-300 ease-in-out hover:scale-105" 
            onClick={onClick}
        >
            <img 
                src={image} 
                alt={t(title)} 
                className="w-36 h-36 rounded-full object-cover shadow-lg mb-4"
            />
            <h3 className="text-lg font-semibold text-gray-800">{t(title)}</h3>
        </div>
    );
};

export default CropCard;
