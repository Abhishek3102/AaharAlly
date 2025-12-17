"use client";

import {
  Card,
  CardHeader,
  CardBody,
  Typography,
  IconButton,
} from "@material-tailwind/react";
import Image from "next/image";
import { useEffect, useState } from "react";
import { FaStar, FaHeart, FaRegHeart } from "react-icons/fa";
import { useRouter, useSearchParams } from "next/navigation";
import axios from "axios";
import { Food } from "@/types";
import Loading from "./loading";
import { useUser } from "@clerk/nextjs";
import toast from "react-hot-toast";

export function BookingCard() {
  const router = useRouter();
  const handleClick = (id: string) => {
    router.push(`/item-details/${id}`);
  };
  const { user } = useUser();

  const searchParams = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [likeLoading, setLikeLoading] = useState(false);
  const [foodArray, setFoodArray] = useState<Food[]>([]);
  const [likedItem, setLikedItem] = useState<string[]>([]);

  // Consolidate fetching logic
  useEffect(() => {
    const controller = new AbortController();
    const signal = controller.signal;
    const categoryParam = searchParams.get("category");
    const regionParam = searchParams.get("region");
    const mealTypeParam = searchParams.get("meal_type");

    const categoriesArray = categoryParam
      ? categoryParam.split(",").map((cat) => cat.trim())
      : [];
    const regionsArray = regionParam
      ? regionParam.split(",").map((reg) => reg.trim())
      : [];

    const isFilterActive =
      categoriesArray.length > 0 ||
      regionsArray.length > 0 ||
      mealTypeParam;

    const fetchData = async () => {
      try {
        setLoading(true);
        // If no filter is active, we still want to fetch data (all data to show in rows)
        // We pass age if available for recommendations, but don't block on it.
        const response = await axios.get(`/api/Users/`, {
          params: {
            categories: categoriesArray.join(","),
            regions: regionsArray.join(","),
            meal_type: mealTypeParam,
            age: user?.unsafeMetadata.age,
          },
          signal,
        });
        setFoodArray(response.data.data);
      } catch (error: any) {
        if (axios.isCancel(error)) {
          console.log("Request canceled:", error.message);
        } else {
          console.error("Error fetching data:", error);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    return () => controller.abort();
  }, [searchParams, user?.unsafeMetadata.age]);

  useEffect(() => {
    const fetchLike = async () => {
        if (!user?.primaryEmailAddress?.emailAddress) return;
      try {
        setLikeLoading(true);
        const resp = await axios.get("/api/fav-food-list/", {
          params: { email: user?.primaryEmailAddress?.emailAddress },
        });
        if (
          resp.data.favoriteFoods.length &&
          resp.data.favoriteFoods.length > 0
        ) {
          setLikedItem(
            resp.data.favoriteFoods.map((m: { _id: string }) => m._id)
          );
        }
      } catch (error: any) {
        toast.error(error.response?.message || "An error occurred");
      } finally {
        setLikeLoading(false);
      }
    };

    if (user) {
        fetchLike();
    }
  }, [user]);

  const likeProduct = async (like: boolean, id: string) => {
    if (!user) {
        toast.error("Please login to like items");
        return;
    }
    try {
      setLikeLoading(true);
      const resp = await axios.post("/api/add-fav-food/", {
        foodId: id,
        like,
        email: user?.primaryEmailAddress?.emailAddress,
      });
      if (
        resp.data.user.favoriteFoods.length &&
        resp.data.user.favoriteFoods.length > 0
      ) {
        setLikedItem(resp.data.user.favoriteFoods);
      }
    } catch (error) {
      console.log({ error });
    } finally {
      setLikeLoading(false);
    }
  };

  const renderCard = (item: Food) => (
    // @ts-expect-error: Material Tailwind types using ref causing conflict with React 19
    <Card
      key={item._id}
      onClick={() => handleClick(item._id)}
      className="group min-w-[300px] w-full max-w-[26rem] shadow-lg sm:max-w-[20rem] md:max-w-[22rem] lg:max-w-[24rem] cursor-pointer hover:scale-105 hover:shadow-blue-gray-300 hover:shadow-lg transition-transform duration-300 ease-in-out snap-center"
      placeholder={"Cards"}
    >
      {/* @ts-expect-error: Material Tailwind types using ref causing conflict with React 19 */}
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
        {/* @ts-expect-error: Material Tailwind types using ref causing conflict with React 19 */}
        <IconButton
          size="sm"
          color="red"
          variant="text"
          disabled={likeLoading}
          className="!absolute top-4 right-4 rounded-full"
          onClick={(e) => {
            e.stopPropagation();
            likeProduct(
              likedItem.find((itm) => itm == item._id) ? false : true,
              item._id
            );
          }}
          placeholder={undefined}
        >
          {(likedItem.find((itm) => itm == item._id) ? true : false) ? (
            <FaHeart className="h-6 w-6 text-red-600" />
          ) : (
            <FaRegHeart className="h-6 w-6" />
          )}
        </IconButton>
      </CardHeader>
      {/* @ts-expect-error: Material Tailwind types using ref causing conflict with React 19 */}
      <CardBody
        placeholder={undefined}
      >
        <div className="mb-1 flex items-center justify-between">
          {/* @ts-expect-error: Material Tailwind types using ref causing conflict with React 19 */}
          <Typography
            variant="h5"
            color="blue-gray"
            className="hover:text-redCustom font-medium truncate"
            placeholder={undefined}
          >
            {item.name}
          </Typography>
          {/* @ts-expect-error: Material Tailwind types using ref causing conflict with React 19 */}
          <Typography
            color="blue-gray"
            className="flex items-center gap-1.5 font-normal"
            placeholder={undefined}
          >
            <FaStar className="text-yellow-500 h-5 w-5" />
            {item.rating}
          </Typography>
        </div>
        {/* @ts-expect-error: Material Tailwind types using ref causing conflict with React 19 */}
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

  const isFilterActive =
    searchParams.get("category") ||
    searchParams.get("region") ||
    searchParams.get("meal_type");

  const categories = ["Breakfast", "Lunch", "Dinner"];

  if (loading) return <Loading />;

  if (!isFilterActive) {
      return (
          <div className="w-full flex flex-col gap-10 pb-10">
              {categories.map((cat) => {
                  const items = foodArray.filter(item => item.meal_type === cat).slice(0, 5);
                  if (items.length === 0) return null;
                  return (
                      <div key={cat} className="flex flex-col gap-4">
                          <div className="flex justify-between items-center px-4">
                              <h2 className="text-2xl font-bold text-gray-800">{cat}</h2>
                              <button 
                                onClick={() => router.push(`?meal_type=${cat}`)}
                                className="text-orangeCustom font-semibold hover:underline"
                              >
                                  View More
                              </button>
                          </div>
                          <div className="flex gap-4 overflow-x-auto pb-6 px-4 snap-x hide-scrollbar">
                              {items.map(renderCard)}
                          </div>
                      </div>
                  );
              })}
          </div>
      );
  }

  return (
    <div className="flex flex-wrap justify-center gap-8">
      {foodArray.length > 0 ? (
          foodArray.map(renderCard)
      ) : (
          <div className="text-center w-full py-10 text-gray-500">No items found</div>
      )}
    </div>
  );
}

export default BookingCard;
