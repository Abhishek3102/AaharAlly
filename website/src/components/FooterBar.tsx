"use client";
import React from "react";
import { FaShoppingCart, FaMapMarkerAlt, FaBell, FaHeart } from "react-icons/fa";
import { IconType } from "react-icons";
import { useRouter } from "next/navigation"; // Use this for App Router

interface FooterBarProps {
  cartItems: number;
  notifications: number;
}

interface FooterButtonProps {
  icon: IconType;
  label: string;
  path: string;
  badge?: number;
  iconColor: string;
}

const FooterBar: React.FC<FooterBarProps> = ({ cartItems, notifications }) => {
  const router = useRouter();

  const buttons: FooterButtonProps[] = [
    {
      icon: FaMapMarkerAlt,
      label: "Explore",
      path: "/explore",
      iconColor: "text-white",
    },
    {
      icon: FaShoppingCart,
      label: "Cart",
      path: "/cart",
      badge: cartItems,
      iconColor: "text-white",
    },
    {
      icon: FaHeart,
      label: "Favorites",
      path: "/favorites",
      iconColor: "text-white",
    },
    {
      icon: FaBell,
      label: "Notifications",
      path: "/notifications",
      badge: notifications,
      iconColor: "text-white",
    },
  ];

  const handleNavigate = (path: string) => {
    router.push(path);
  };

  return (
    <div className="fixed bottom-0 flex w-full bg-whitescale-105 justify-center z-50 bg-transparent lg:hidden">
      <div className="fixed bottom-0 w-[40vw] flex justify-around items-center rounded-3xl px-4 mx-2 my-4 z-50 bg-redCustom text-white shadow-lg shadow-black transition duration-500 bg-opacity-95">
        {buttons.map((button, index) => (
          <div
            key={index}
            className="relative cursor-pointer p-2 flex items-center rounded-full transition-all"
            onClick={() => handleNavigate(button.path)}
          >
            <button.icon className={`text-xl ${button.iconColor} hover:text-yellow-400`} />
            {button.badge && button.badge > 0 && (
              <span className="absolute -top-1 -right-2 bg-red-600 text-black rounded-full text-xs px-2">
                {button.badge}
              </span>
            )}
            <span className="text-sm mt-1 text-black hidden md:flex justify-center items-center md:ml-2">{button.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FooterBar;
