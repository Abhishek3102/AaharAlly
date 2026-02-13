'use client';
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

const AdminRestaurants = () => {
    const [hotels, setHotels] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const router = useRouter();

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
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

    if (loading) return <div className="p-8 text-center">Loading Hotels...</div>;

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <div className="max-w-7xl mx-auto">
                <div className="flex justify-between items-center mb-8">
                    <h1 className="text-3xl font-bold text-gray-800">Manage Restaurants</h1>
                    <div className="flex gap-4">
                         <button onClick={() => router.push('/admin/dashboard')} className="text-blue-600 hover:underline">
                            ← Back to Dashboard
                        </button>
                    </div>
                </div>

                <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                    <table className="w-full text-left">
                        <thead className="bg-gray-50 border-b border-gray-200">
                            <tr>
                                <th className="p-4 font-semibold text-gray-600">Restaurant</th>
                                <th className="p-4 font-semibold text-gray-600">Location</th>
                                <th className="p-4 font-semibold text-gray-600">Owner</th>
                                <th className="p-4 font-semibold text-gray-600">Rating</th>
                                <th className="p-4 font-semibold text-gray-600">Menu Items</th>
                                <th className="p-4 font-semibold text-gray-600">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                            {hotels.map((hotel) => (
                                <tr key={hotel._id} className="hover:bg-gray-50 transition">
                                    <td className="p-4 flex items-center gap-3">
                                        <img src={hotel.image} alt={hotel.name} className="w-12 h-12 rounded object-cover" />
                                        <span className="font-bold text-gray-800">{hotel.name}</span>
                                    </td>
                                    <td className="p-4 text-gray-600 max-w-xs truncate" title={hotel.address}>
                                        {hotel.address}
                                    </td>
                                    <td className="p-4">
                                        <div className="flex items-center gap-2">
                                            <img src={hotel.owner.image} alt="Owner" className="w-8 h-8 rounded-full" />
                                            <span className="text-sm">{hotel.owner.name}</span>
                                        </div>
                                    </td>
                                    <td className="p-4 font-bold text-yellow-600">
                                        ⭐ {hotel.rating}
                                    </td>
                                    <td className="p-4 text-gray-600">
                                        {hotel.menu ? hotel.menu.length : 0} Items
                                    </td>
                                    <td className="p-4">
                                        <Link href={`/hotels/${hotel._id}`} className="text-blue-500 hover:text-blue-700 font-medium text-sm">
                                            View Live
                                        </Link>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default AdminRestaurants;
