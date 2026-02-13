'use client';

import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams } from 'next/navigation';
import SingleCard from '@/components/SingleCard'; // Corrected Import
import Link from 'next/link';

const HotelDetailsPage = () => {
    const { id } = useParams();
    const [hotel, setHotel] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!id) return;
        const fetchHotel = async () => {
            try {
                const res = await axios.get(`/api/restaurants/${id}`);
                if (res.data.success) {
                    setHotel(res.data.restaurant);
                }
            } catch (error) {
                console.error("Failed to fetch hotel details", error);
            } finally {
                setLoading(false);
            }
        };
        fetchHotel();
    }, [id]);

    if (loading) return <div className="min-h-screen pt-24 text-center">Loading Hotel Details...</div>;
    if (!hotel) return <div className="min-h-screen pt-24 text-center">Hotel Not Found</div>;

    return (
        <div className="min-h-screen bg-gray-50 pt-20">
            {/* Hero Section */}
            <div className="relative h-[400px]">
                <img src={hotel.image} alt={hotel.name} className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex items-end">
                    <div className="container mx-auto px-4 md:px-10 pb-10 text-white">
                        <h1 className="text-5xl font-bold mb-2">{hotel.name}</h1>
                        <p className="text-xl opacity-90 mb-4 max-w-2xl">{hotel.description}</p>
                        <div className="flex flex-wrap gap-4 items-center">
                            <span className="bg-green-500 px-3 py-1 rounded font-bold">⭐ {hotel.rating}</span>
                            <span className="flex items-center gap-1">📍 {hotel.address}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="container mx-auto px-4 md:px-10 py-12 pb-20 flex flex-col lg:flex-row gap-16 items-start">
                {/* Main Content: Menu */}
                <div className="flex-1 w-full">
                    <h2 className="text-3xl font-bold text-gray-800 mb-8">Exclusive Menu</h2>
                    <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-8">
                        {hotel.menu && hotel.menu.length > 0 ? (
                            hotel.menu.map((food: any) => {
                                if (!food) return null;
                                return <SingleCard key={food._id} item={food} />;
                            })
                        ) : (
                            <p className="text-gray-500 italic">No menu items available.</p>
                        )}
                    </div>
                </div>

                {/* Sidebar: Owner & Info */}
                <div className="w-full lg:w-1/3">
                    <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100 sticky top-24">
                        <h3 className="text-xl font-bold text-gray-800 mb-6 border-b pb-2">Hotel Owner</h3>
                        <div className="flex items-center gap-4 mb-6">
                            <img src={hotel.owner.image} alt={hotel.owner.name} className="w-20 h-20 rounded-full object-cover border-2 border-redCustom" />
                            <div>
                                <p className="text-lg font-bold text-gray-800">{hotel.owner.name}</p>
                                <p className="text-sm text-gray-500">Owner & Operator</p>
                            </div>
                        </div>
                        <p className="text-gray-600 text-sm italic mb-6">
                            "Welcome to {hotel.name}. We pride ourselves on delivering the finest culinary experiences with authentic flavors and world-class hospitality."
                        </p>
                        
                        <div className="mt-6">
                             <a 
                                href={`https://www.google.com/maps/search/?api=1&query=${hotel.location.lat},${hotel.location.lng}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block w-full text-center bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition"
                             >
                                Get Directions 🗺️
                             </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HotelDetailsPage;
