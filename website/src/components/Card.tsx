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
  const [initialLoading, setInitialLoading] = useState(true); // Loading for first fetch
  const [loading, setLoading] = useState(false); // Loading for server-side refetch
  const [likeLoading, setLikeLoading] = useState(false);
  const [allFood, setAllFood] = useState<Food[]>([]); // Store ALL data
  const [filteredFood, setFilteredFood] = useState<Food[]>([]); // Displayed data
  const [likedItem, setLikedItem] = useState<string[]>([]);

  // 1. Fetch ALL data on mount (Optimized)
  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setInitialLoading(true);
        const response = await axios.get(`/api/Users/`); // Fetch everything
        if (response.data.success) {
            setAllFood(response.data.data);
            setFilteredFood(response.data.data);
        }
      } catch (error) {
        console.error("Error fetching initial data:", error);
      } finally {
        setInitialLoading(false);
      }
    };
    fetchAllData();
  }, []);

  // 2. Client-Side Filtering (Instant)
  // We only re-fetch from server if strictly necessary (like Region logic which is complex).
  // For Category and Meal Type, we filter locally.
  useEffect(() => {
    if (allFood.length === 0) return;

    const categoryParam = searchParams.get("category");
    const regionParam = searchParams.get("region");
    const mealTypeParam = searchParams.get("meal_type");
    const searchParam = searchParams.get("search");
    const healthConditionParam = searchParams.get("health_condition");

    // If Region OR Health Condition is present, we used Server Logic.
    // Region: Complex Aggregation.
    // Health Condition: Gemini AI Filtering.
    if (regionParam || healthConditionParam) {
       const fetchServerData = async () => {
          setLoading(true);
          try {
             // Pass all params to server
             const response = await axios.get(`/api/Users/`, {
                params: {
                   categories: categoryParam,
                   regions: regionParam,
                   meal_type: mealTypeParam,
                   search: searchParam,
                   health_condition: healthConditionParam 
                }
             });
             setFilteredFood(response.data.data);
          } catch(err) {
             console.error(err);
          } finally {
             setLoading(false);
          }
       };
       fetchServerData();
       return; 
    }

    // --- INSTANT CLIENT FILTERING ---
    // Only applied if we rely on standard category/meal_type filtering
    let result = [...allFood];

    if (categoryParam) {
        const cats = categoryParam.split(",").map(c => c.trim().toLowerCase());
        result = result.filter(item => cats.includes(item.category.toLowerCase()));
    }

    if (mealTypeParam) {
        result = result.filter(item => item.meal_type.toLowerCase() === mealTypeParam.toLowerCase());
    }
    
    // Simple client-side search if needed (optional)
    if (searchParam) {
       // logic if search param was passed (though Navbar usually handles search URL)
    }

    setFilteredFood(result);

  }, [searchParams, allFood]); 


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
        if(error?.response?.data?.message) toast.error(error.response.data.message);
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
        {/* REPLACED Next.js Image with standard img for reliability */}
        <Image
          src={item.image}
          alt={item.name}
          className="object-cover group-hover:scale-110 group-hover:shadow-xl transition-transform duration-300 ease-in-out h-full w-full"
          width={500}
          height={500}
          unoptimized={true}
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
    searchParams.get("search") || 
    searchParams.get("meal_type");

  const categories = ["Breakfast", "Lunch", "Dinner"];

  if (initialLoading || loading) return <Loading />;

  const healthCondition = searchParams.get("health_condition");

  if (!isFilterActive) {
      return (
          <div className="w-full flex flex-col gap-10 pb-10">
              {categories.map((cat) => {
                  // Use filteredFood so server-side filters (Health) apply
                  const items = filteredFood.filter(item => item.meal_type === cat).slice(0, 5);
                  if (items.length === 0) return null;
                  return (
                      <div key={cat} className="flex flex-col gap-4">
                          <div className="flex justify-between items-center px-4">
                              <h2 className="text-2xl font-bold text-gray-800">
                                {healthCondition ? `${cat} Suggestion for ${healthCondition}` : cat}
                              </h2>
                              <button 
                                onClick={() => router.push(`?meal_type=${cat}${healthCondition ? `&health_condition=${healthCondition}` : ''}`)}
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
      {filteredFood.length > 0 ? (
          filteredFood.map(renderCard)
      ) : (
          <div className="text-center w-full py-10 text-gray-500">No items found</div>
      )}
    </div>
  );
}

export default BookingCard;
