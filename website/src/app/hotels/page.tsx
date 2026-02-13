'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import axios from 'axios';
import Link from 'next/link';

// Dynamically import Map (No SSR)
const MapComponent = dynamic(() => import('@/components/MapComponent'), {
    ssr: false,
    loading: () => <div className="h-[500px] w-full bg-gray-200 animate-pulse rounded-lg flex items-center justify-center">Loading Map...</div>
});

const HotelsPage = () => {
    const [hotels, setHotels] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHotels = async () => {
            try {
                const res = await axios.get('/api/restaurants');
                if (res.data.success) {
                    setHotels(res.data.restaurants);
                }
            } catch (error) {
                console.error("Failed to fetch hotels", error);
            } finally {
                setLoading(false);
            }
        };
        fetchHotels();
    }, []);

    if (loading) return <div className="min-h-screen pt-24 text-center">Loading Hotels...</div>;

    return (
        <div className="min-h-screen bg-gray-50 pt-24 pb-12 px-4 md:px-10">
            <div className="container mx-auto">
                <h1 className="text-4xl font-bold text-gray-800 mb-2">Explore Premium Hotels</h1>
                <p className="text-gray-600 mb-8">Discover top-rated dining destinations across Mumbai.</p>

                {/* Map Section */}
                <div className="mb-12">
                     <MapComponent hotels={hotels} />
                </div>

                {/* List Section */}
                <h2 className="text-2xl font-bold text-gray-800 mb-6">All Hotels</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {hotels.map((hotel) => (
                        <Link href={`/hotels/${hotel._id}`} key={hotel._id} className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition duration-300 block group">
                            <div className="relative h-48 overflow-hidden">
                                <img src={hotel.image} alt={hotel.name} className="w-full h-full object-cover group-hover:scale-105 transition duration-500" />
                                <div className="absolute top-2 right-2 bg-white px-2 py-1 rounded-md text-sm font-bold shadow">
                                    ⭐ {hotel.rating}
                                </div>
                            </div>
                            <div className="p-6">
                                <h3 className="text-xl font-bold text-gray-800 mb-1 group-hover:text-redCustom transition">{hotel.name}</h3>
                                <p className="text-gray-500 text-sm mb-4 line-clamp-2">{hotel.description}</p>
                                
                                <div className="flex items-center gap-3 mb-4">
                                     <img src={hotel.owner.image} alt={hotel.owner.name} className="w-8 h-8 rounded-full object-cover border border-gray-200" />
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
                    ))}
                </div>
            </div>
        </div>
    );
};

export default HotelsPage;
