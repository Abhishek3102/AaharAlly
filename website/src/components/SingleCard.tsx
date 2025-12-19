
"use client";

import {
  Card,
  CardHeader,
  CardBody,
  Typography,
  IconButton,
} from "@material-tailwind/react";
import Image from "next/image";
import { FaStar, FaHeart, FaRegHeart } from "react-icons/fa";
import { useRouter } from "next/navigation";
import { useState } from "react";
import toast from "react-hot-toast";
import { useUser } from "@clerk/nextjs";
import axios from "axios";

export default function SingleCard({ item }: { item: any }) {
  const router = useRouter();
  const { user } = useUser();
  const [likeLoading, setLikeLoading] = useState(false);
  const [isLiked, setIsLiked] = useState(false); // Simplified local state for now
  
  // Real implementation should check initial like status from a global context or prop
  // For now, we start unliked or maybe we can fetch?
  // To avoid complexity, let's keep it simple: click navigates, like is optimistic.

  const handleClick = () => {
    router.push(`/item-details/${item._id}`);
  };

  const handleLike = async (e: any) => {
    e.stopPropagation();
    if (!user) {
        toast.error("Please login to like");
        return;
    }
    setLikeLoading(true);
    try {
        await axios.post("/api/add-fav-food/", {
            foodId: item._id,
            like: !isLiked,
            email: user?.primaryEmailAddress?.emailAddress,
        });
        setIsLiked(!isLiked);
        toast.success(isLiked ? "Removed from favorites" : "Added to favorites");
    } catch (err) {
        toast.error("Failed to update favorite");
    } finally {
        setLikeLoading(false);
    }
  };

  return (
    // @ts-expect-error: Material Tailwind type conflict
    <Card
      onClick={handleClick}
      className="group min-w-[300px] w-full max-w-[26rem] shadow-lg sm:max-w-[20rem] md:max-w-[22rem] lg:max-w-[24rem] cursor-pointer hover:scale-105 hover:shadow-blue-gray-300 hover:shadow-lg transition-transform duration-300 ease-in-out"
      placeholder={undefined}
    >
      {/* @ts-expect-error: Material Tailwind type conflict */}
      <CardHeader
        color="blue-gray"
        className="relative h-56"
        floated={false}
        placeholder={undefined}
      >
        <div className="absolute top-4 left-4 z-10">
          <div className="!rounded-full bg-peachCustom bg-opacity-85 px-2 py-1 text-white text-xs md:text-sm">
            <div className="flex justify-start items-center text-redCustom">
              <span className="text-sm md:text-base">
                ⭐ {item.rating}
              </span>
            </div>
          </div>
        </div>
        <Image
          src={item.image}
          alt={item.name}
          className="object-cover group-hover:scale-110 group-hover:shadow-xl transition-transform duration-300 ease-in-out h-full w-full"
          loading="lazy"
          width={500}
          height={500}
        />
        <div className="absolute inset-0 h-full w-full bg-gradient-to-tr from-transparent via-transparent to-black/60" />
        {/* @ts-expect-error: Material Tailwind type conflict */}
        <IconButton
          size="sm"
          color="red"
          variant="text"
          disabled={likeLoading}
          className="!absolute top-4 right-4 rounded-full"
          onClick={handleLike}
          placeholder={undefined}
        >
          {isLiked ? (
            <FaHeart className="h-6 w-6 text-red-600" />
          ) : (
            <FaRegHeart className="h-6 w-6" />
          )}
        </IconButton>
      </CardHeader>
      {/* @ts-expect-error: Material Tailwind type conflict */}
      <CardBody placeholder={undefined}>
        <div className="mb-1 flex items-center justify-between">
          {/* @ts-expect-error: Material Tailwind type conflict */}
          <Typography
            variant="h5"
            color="blue-gray"
            className="hover:text-redCustom font-medium truncate"
            placeholder={undefined}
          >
            {item.name}
          </Typography>
          {/* @ts-expect-error: Material Tailwind type conflict */}
          <Typography
            color="blue-gray"
            className="flex items-center gap-1.5 font-normal"
            placeholder={undefined}
          >
            <FaStar className="text-yellow-500 h-5 w-5" />
            {item.rating}
          </Typography>
        </div>
        {/* @ts-expect-error: Material Tailwind type conflict */}
        <Typography
            color="gray"
            className="line-clamp-2"
            placeholder={undefined}
        >
            {item.description}
        </Typography>
      </CardBody>
    </Card>
  );
}
