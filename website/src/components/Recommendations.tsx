
"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import SingleCard from "./SingleCard";
import { toast } from "react-hot-toast";

interface FoodItem {
    _id: string;
    name: string;
    description: string;
    price: string;
    image: string;
    category: string;
    meal_type: string;
    rating: string;
}

const Recommendations = () => {
    const [items, setItems] = useState<FoodItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [meta, setMeta] = useState<any>(null);

    useEffect(() => {
        const fetchRecommendations = async () => {
            try {
                const res = await axios.get("/api/recommendations");
                if (res.data.action === "complete_profile") {
                    // Profile not complete, component can just hide or show a specific message
                    // (The Modal handles the prompt usually)
                    setLoading(false);
                    return;
                }
                
                if (res.data.success) {
                    setItems(res.data.data);
                    setMeta(res.data.meta);
                }
            } catch (error: any) {
                console.error("Failed to load recommendations", error);
                // Don't show toast on 401/unauth to avoid spam if user logged out
                // toast.error("Could not load recommendations");
            } finally {
                setLoading(false);
            }
        };

        fetchRecommendations();
    }, []);

    if (loading) return <div className="text-center py-8">Loading recommendations...</div>;
    if (!items || items.length === 0) return null;

    return (
        <div className="w-full max-w-7xl mx-auto py-8 px-4">
             <div className="mb-6">
                <h2 className="text-3xl font-bold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
                    Recommended For You 😋
                </h2>
                {meta?.categories && (
                    <p className="text-gray-500 text-sm mt-1">
                        Based on your taste for: <span className="font-medium text-gray-700">{meta.categories.slice(0, 3).join(", ")}...</span>
                    </p>
                )}
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                {items.map((item) => (
                    <SingleCard key={item._id} item={item} />
                ))}
            </div>
        </div>
    );
};

export default Recommendations;
