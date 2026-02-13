'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import axios from 'axios';
import Link from 'next/link';
import Image from 'next/image';

// Dynamically import Map (No SSR)
const MapComponent = dynamic(() => import('@/components/MapComponent'), {
    ssr: false,
    loading: () => <div className="h-[500px] w-full bg-gray-200 animate-pulse rounded-lg flex items-center justify-center">Loading Map...</div>
});

const HotelsPage = () => {
    const [hotels, setHotels] = useState<any[]>([]);
    const [filteredHotels, setFilteredHotels] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const [isSearching, setIsSearching] = useState(false);

    useEffect(() => {
        const fetchHotels = async () => {
            try {
                const res = await axios.get('/api/restaurants');
                if (res.data.success) {
                    setHotels(res.data.restaurants);
                    setFilteredHotels(res.data.restaurants);
                }
            } catch (error) {
                console.error("Failed to fetch hotels", error);
            } finally {
                setLoading(false);
            }
        };
        fetchHotels();
    }, []);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!searchQuery.trim()) {
            setFilteredHotels(hotels);
            return;
        }

        setIsSearching(true);
        try {
            // Use the existing Vector Search API
            const res = await axios.get(`/api/Users?search=${encodeURIComponent(searchQuery)}`);
            if (res.data.success) {
                const matchingFoods = res.data.data;
                const matchingFoodIds = new Set(matchingFoods.map((f: any) => f._id));

                // Filter hotels that have AT LEAST one matching food item
                const filtered = hotels.filter(hotel => 
                    hotel.menu.some((menuItem: any) => matchingFoodIds.has(menuItem._id))
                );
                setFilteredHotels(filtered);
            }
        } catch (error) {
            console.error("Search failed", error);
        } finally {
            setIsSearching(false);
        }
    };

    const handleClear = () => {
        setSearchQuery("");
        setFilteredHotels(hotels);
    };

    if (loading) return <div className="min-h-screen pt-24 text-center">Loading Hotels...</div>;

    return (
        <div className="min-h-screen bg-gray-50 pt-24 pb-12 px-4 md:px-10">
            <div className="container mx-auto">
                <h1 className="text-4xl font-bold text-gray-800 mb-2">Explore Premium Hotels</h1>
                <p className="text-gray-600 mb-8">Discover top-rated dining destinations across Mumbai.</p>

                <p className="text-gray-600 mb-8">Discover top-rated dining destinations across Mumbai.</p>

                {/* Search Bar */}
                <form onSubmit={handleSearch} className="mb-8 flex gap-2 max-w-2xl">
                    <input
                        type="text"
                        placeholder="Search for food (e.g., 'Spicy paneer', 'Vegan options')..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="flex-1 p-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-orangeCustom"
                    />
                    <button 
                        type="submit" 
                        disabled={isSearching}
                        className="bg-orangeCustom text-white px-6 py-3 rounded-lg font-semibold hover:bg-orange-600 transition disabled:opacity-50"
                    >
                        {isSearching ? 'Searching...' : 'Search'}
                    </button>
                    {searchQuery && (
                         <button 
                            type="button" 
                            onClick={handleClear}
                            className="bg-gray-200 text-gray-700 px-4 py-3 rounded-lg font-semibold hover:bg-gray-300 transition"
                        >
                            Clear
                        </button>
                    )}
                </form>

                {/* Map Section */}
                <div className="mb-12">
                     <MapComponent hotels={filteredHotels} />
                </div>

                {/* List Section */}
                <h2 className="text-2xl font-bold text-gray-800 mb-6">
                    {searchQuery ? `Search Results (${filteredHotels.length})` : 'All Hotels'}
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {filteredHotels.length > 0 ? (
                        filteredHotels.map((hotel) => (
                        <Link href={`/hotels/${hotel._id}`} key={hotel._id} className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition duration-300 block group">
                            <div className="relative h-48 overflow-hidden">
                                <Image 
                                    src={hotel.image} 
                                    alt={hotel.name} 
                                    fill
                                    className="object-cover group-hover:scale-105 transition duration-500"
                                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                                />
                                <div className="absolute top-2 right-2 bg-white px-2 py-1 rounded-md text-sm font-bold shadow z-10">
                                    ⭐ {hotel.rating}
                                </div>
                            </div>
                            <div className="p-6">
                                <h3 className="text-xl font-bold text-gray-800 mb-1 group-hover:text-redCustom transition">{hotel.name}</h3>
                                <p className="text-gray-500 text-sm mb-4 line-clamp-2">{hotel.description}</p>
                                
                                <div className="flex items-center gap-3 mb-4">
                                     <Image 
                                        src={hotel.owner.image} 
                                        alt={hotel.owner.name} 
                                        width={32} 
                                        height={32}
                                        className="rounded-full object-cover border border-gray-200" 
                                    />
                                     <div>
                                         <p className="text-xs text-gray-400 uppercase font-semibold">Owner</p>
                                         <p className="text-sm font-medium text-gray-700">{hotel.owner.name}</p>
                                     </div>
                                </div>

                                <div className="text-sm text-gray-500 flex items-start gap-1">
                                    <span>📍</span>
                                    <span className="truncate">{hotel.address}</span>
                                </div>
                            </div>
                        </Link>
                    ))
                    ) : (
                        <div className="col-span-full text-center py-10 bg-white rounded-xl shadow p-8">
                            <p className="text-xl text-gray-500 mb-2">No hotels found matching "{searchQuery}"</p>
                            <p className="text-gray-400">Try searching for different food items.</p>
                            <button onClick={handleClear} className="mt-4 text-orangeCustom underline font-semibold">Clear Search</button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default HotelsPage;
