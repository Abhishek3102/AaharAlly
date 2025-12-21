"use client";
// import AgeModal from "@/components/AgeModal"; // Adjust the path as necessary
import { BannerCarousel } from "@/components/BannerCarousel";
import FilterComponent from "@/components/Filter";
import FoodItemCard from "@/components/FoodItemCard";
import Card from "@/components/Card";
import Recommendations from "@/components/Recommendations";
import { Suspense } from "react";

import { useSearchParams } from "next/navigation";

const ExploreContent = () => {
    const searchParams = useSearchParams();
    const search = searchParams.get("search");

    if (search) {
        return (
            <div>
                 <main className="flex flex-col gap-8 items-center max w-full justify-center mx-auto px-4 sm:px-6 md:px-8 overflow-hidden pt-10">
                    <div className="w-full max-w-7xl">
                        <h1 className="text-3xl font-bold text-gray-800 mb-6">Results for <span className="text-redCustom">"{search}"</span></h1>
                        <Card />
                    </div>
                </main>
            </div>
        );
    }

    return (
        <div>
            {/* Navbar is in global layout */}

            {/* <AgeModal /> Include the Age Modal here */}
            <main className="flex flex-col gap-8 items-center max w-full justify-center mx-auto px-4 sm:px-6 md:px-8 overflow-hidden">
                <BannerCarousel />
                <Recommendations />
                <FilterComponent />
                <FoodItemCard />
                <Card />
            </main>
        </div>
    )
}

const Page = () => {
    return (
        <Suspense fallback={<div className="text-center py-20">Loading...</div>}>
            <ExploreContent />
        </Suspense>
    )
}

export default Page