"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
import { useUser } from "@clerk/nextjs";
import Card from "@/components/HomeCard";
import toast from "react-hot-toast";

const FavoritesPage = () => {
  const { user, isLoaded } = useUser();
  const [favorites, setFavorites] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFavorites = async () => {
      if (!isLoaded || !user) return;

      try {
        setLoading(true);
        // Assuming the API returns a list of favorite food objects
        const resp = await axios.get("/api/fav-food-list/", {
          params: { email: user.primaryEmailAddress?.emailAddress },
        });

        // The API returns { favoriteFoods: [...] }
        setFavorites(resp.data.favoriteFoods || []);
      } catch (error) {
        console.error("Error fetching favorites:", error);
        toast.error("Failed to load favorites.");
      } finally {
        setLoading(false);
      }
    };

    fetchFavorites();
  }, [isLoaded, user]);

  if (!isLoaded) return <div className="text-center py-20">Loading...</div>;
  if (!user) return <div className="text-center py-20">Please sign in to view your favorites.</div>;

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">My Favorites</h1>
      
      {loading ? (
        <div className="text-center py-10">Loading your favorites...</div>
      ) : favorites.length === 0 ? (
        <div className="text-center py-10 text-gray-500">
          You haven&apos;t added any favorites yet.
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
          {favorites.map((item: any, index: number) => (
            <Card
              key={item._id || index}
              title={item.name}
              description={item.description}
              imageSrc={item.image}
              // You might need to pass other props depending on your Card component
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default FavoritesPage;
