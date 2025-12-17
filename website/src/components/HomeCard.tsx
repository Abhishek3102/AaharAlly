import React from "react";

interface HomeCardProps {
  title: string;
  description: string;
  imageSrc: string;
}

const Card: React.FC<HomeCardProps> = ({ title, description, imageSrc }) => {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden w-80">
      <img
        src={imageSrc}
        alt={title}
        className="w-full h-auto object-contain"
      />
      <div className="p-4">
        <h3 className="font-semibold text-lg text-gray-800">{title}</h3>
        <p className="text-gray-600">{description}</p>
      </div>
    </div>
  );
};

export default Card;
