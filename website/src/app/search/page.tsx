"use client";

import React, { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Card,
  CardHeader,
  CardBody,
  Typography,
  IconButton,
} from "@material-tailwind/react";
import Image from "next/image";
import { FaStar } from "react-icons/fa";

interface FoodItem {
  _id: string;
  name: string;
  description: string;
  price: string;
  image: string;
  category: string;
  meal_type: string;
  score?: number;
  rating?: number;
}

const SearchContent = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const query = searchParams.get("search");
  const [results, setResults] = useState<FoodItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchResults = async () => {
      if (!query) return;

      setLoading(true);
      try {
        const res = await fetch(`/api/Users?search=${encodeURIComponent(query)}`);
        const data = await res.json();

        if (data.success) {
          setResults(data.data);
          
          console.group(`🔍 Search Results for: "${query}"`);
          data.data.forEach((item: FoodItem, index: number) => {
            if (item.score) {
                console.log(`#${index + 1} ${item.name}: Score ${item.score.toFixed(4)}`);
            } else {
                console.log(`#${index + 1} ${item.name}: (No Score / Regex Match)`);
            }
          });
          console.groupEnd();
        }
      } catch (error) {
        console.error("Search failed:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [query]);

  const handleClick = (id: string) => {
    router.push(`/item-details/${id}`);
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-24 px-4 pb-10">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-6">
          Results for <span className="text-orange-500">"{query}"</span>
        </h1>

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-orange-500"></div>
          </div>
        ) : results.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {results.map((item) => (
                // @ts-expect-error: Material Tailwind types conflict
                <Card
                key={item._id}
                onClick={() => handleClick(item._id)}
                className="group w-full shadow-lg cursor-pointer hover:scale-105 hover:shadow-xl transition-transform duration-300 ease-in-out"
                placeholder={undefined}
                >
                {/* @ts-expect-error: Material Tailwind types conflict */}
                <CardHeader
                    color="blue-gray"
                    className="relative h-56"
                    floated={false}
                    placeholder={undefined}
                >
                    <div className="absolute top-4 left-4 z-10">
                    <div className="!rounded-full bg-orange-500 bg-opacity-90 px-2 py-1 text-white text-xs font-bold">
                         ⭐ {item.rating || 4.5}
                    </div>
                    </div>
                    {/* Use standard img for simplicity/robustness in search grid */}
                    <img
                    src={item.image}
                    alt={item.name}
                    className="object-cover h-full w-full group-hover:scale-110 transition-transform duration-500"
                    />
                     <div className="absolute inset-0 h-full w-full bg-gradient-to-tr from-transparent via-transparent to-black/60" />
                </CardHeader>
                {/* @ts-expect-error: Material Tailwind types conflict */}
                <CardBody placeholder={undefined}>
                    <div className="mb-2 flex items-center justify-between">
                    {/* @ts-expect-error: Material Tailwind types conflict */}
                    <Typography variant="h5" color="blue-gray" className="font-medium truncate" placeholder={undefined}>
                        {item.name}
                    </Typography>
                    {/* @ts-expect-error: Material Tailwind types conflict */}
                    <Typography color="blue-gray" className="font-medium" placeholder={undefined}>
                        {item.price}
                    </Typography>
                    </div>
                    {/* @ts-expect-error: Material Tailwind types conflict */}
                    <Typography color="gray" className="line-clamp-2 text-sm" placeholder={undefined}>
                        {item.description}
                    </Typography>
                </CardBody>
                </Card>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-500 mt-20">
            <p className="text-xl">No results found for "{query}".</p>
          </div>
        )}
      </div>
    </div>
  );
};

const SearchPage = () => {
    return (
        <Suspense fallback={
            <div className="flex justify-center items-center h-screen">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-orange-500"></div>
            </div>
        }>
            <SearchContent />
        </Suspense>
    );
};

export default SearchPage;
