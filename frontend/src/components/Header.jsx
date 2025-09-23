import React from "react";
import { useTranslation } from 'react-i18next';

const Header = ({ name }) => {
  const { t } = useTranslation();

  return (
    <div className="h-[40vh] bg-gradient-to-t from-green-600 to-green-400 flex flex-col md:flex-row items-center justify-between px-5 md:px-20 py-8">
      {/* Heading on the left */}
      <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-5xl font-bold text-gray-800 text-center md:text-left">
        {t(name)}
      </h1>

      {/* GIF on the right */}
      <iframe
        src="https://giphy.com/embed/R0MhP0LX2DppguK8fu"
        width="200"
        height="200"
        className="max-w-full h-auto mt-4 md:mt-0 md:w-[400px] md:h-[400px] lg:w-[280px] lg:h-[280px]"
        style={{ border: 0}}
        frameBorder="0"
        allowFullScreen
        title="giphy"
      ></iframe>
    </div>
  );
}

export default Header;
