"use client";
import { useSearchParams } from "next/navigation";
// import Link from "next/link";
// import Image from "next/image";
import Carousel from "@/components/HomeCarousel";
import Card from "@/components/HomeCard";
import HeroCarousel from "@/components/HeroCarousel";
import axios from "axios";
import { useEffect, useState } from "react";

interface FoodItem {
  _id: string;
  name: string;
  description: string;
  image: string;
  category: string;
  meal_type: string;
  price: number;
  rating: number;
  // Add other fields as necessary
}

import { Suspense } from "react";

export default function Home() {
    return (
        <Suspense fallback={<div className="text-center py-20">Loading...</div>}>
            <HomeContent />
        </Suspense>
    );
}

function HomeContent() {
  const [loading, setLoading] = useState(false);
  const [foodArray, setFoodArray] = useState<FoodItem[]>([]);

  const searchParams = useSearchParams();
  const searchText = searchParams.get('search');

  // Assuming foodArray contains the necessary data for rendering
  const featuredItems = foodArray.slice(0, 5); // Adjust slicing as per your data structure
  const popularChoices = foodArray.slice(5, 13); // Adjust slicing as per your data structure

  useEffect(() => {
    setLoading(true);
    const fetchFood = async () => {
      try {
        const resp = await axios.get("/api/Users/", {
            params: { search: searchText || "" },
        });
        setFoodArray(resp.data.data);
      } catch (error) {
        console.error("Error fetching initial data:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchFood();
  }, [searchText]);

  return (
    <>
      <HeroCarousel />
      {loading && <div className="text-center py-10">Loading...</div>}

      <div className="relative z-10 pt-10 px-6 lg:px-12 bg-gray-50">
        <section className="my-10">
          <h2 className="text-3xl font-bold mb-6 text-gray-800">Featured Items</h2>
          <Carousel>
            <div className="flex gap-6">
              {featuredItems.map((item, index) => (
                <Card
                  key={index}
                  title={item.name} // Adjust based on your data structure
                  description={item.description} // Adjust based on your data structure
                  imageSrc={item.image}
                />
              ))}
            </div>
          </Carousel>
        </section>

        <section className="my-10">
          <h2 className="text-3xl font-bold mb-6 text-gray-800">Popular Choices</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {popularChoices.map((item, index) => (
              <Card
                key={index}
                title={item.name} // Adjust based on your data structure
                description={item.description} // Adjust based on your data structure
                imageSrc={item.image} // Adjust based on your data structure
              />
            ))}
          </div>
        </section>
      </div>
    </>
  );
}
