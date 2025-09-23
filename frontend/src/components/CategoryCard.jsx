const CategoryCard = ({ category, details, onClick }) => {
    return (
        <div 
            className="relative w-full max-w-4xl mx-auto bg-gradient-to-r from-green-700 to-yellow-700 text-white 
                        rounded-2xl shadow-lg p-6 sm:p-8 lg:p-10 transition-transform duration-300 
                        hover:scale-[1.02] hover:shadow-2xl cursor-pointer"
            onClick={onClick}
        >
            {/* Glow Effect */}
            <div className="absolute inset-0 bg-white opacity-10 blur-lg rounded-2xl"></div>

            {/* Content */}
            <div className="relative z-10">
                <h3 className="text-2xl sm:text-3xl font-bold mb-4">{category}</h3>
                <p className="text-md sm:text-lg">{details || "Click to see more details!"}</p>
            </div>
        </div>
    );
};

export default CategoryCard;
