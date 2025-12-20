import React from "react";

interface HomeCardProps {
  title: string;
  description: string;
  imageSrc: string;
}

const Card: React.FC<HomeCardProps & { className?: string }> = ({ title, description, imageSrc, className = "" }) => {
  return (
    <div className={`bg-white rounded-lg shadow-md overflow-hidden flex flex-col ${className}`}>
      <div className="h-48 overflow-hidden">
        <img
            src={imageSrc}
            alt={title}
            className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
        />
      </div>
      <div className="p-4 flex flex-col flex-1">
        <h3 className="font-semibold text-lg text-gray-800 mb-2 truncate" title={title}>{title}</h3>
        <p className="text-gray-600 line-clamp-3 text-sm flex-1">{description}</p>
      </div>
    </div>
  );
};

export default Card;
