
"use client";
import React, { useEffect, useState } from 'react';
import axios from 'axios';
// Assuming you have a Card or similar, but for orders we might want a list view.
// I'll create a clean Tailwind layout.
import Link from 'next/link';

interface OrderItem {
    foodId: {
        name: string;
        image: string;
        price: string;
    };
    quantity: number;
    price: number;
}

interface Order {
    _id: string;
    createdAt: string;
    totalAmount: number;
    status: string;
    items: OrderItem[];
}

import { useUser } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';

const OrdersPage = () => {
    const { user, isLoaded } = useUser();
    const router = useRouter();
    const [orders, setOrders] = useState<Order[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (isLoaded && !user) {
            router.push('/sign-in'); // Redirect if not logged in
            return;
        }

        const fetchOrders = async () => {
            if (!user) return; // double check
            try {
                const res = await axios.get('/api/orders');
                if (res.data.success) {
                    setOrders(res.data.orders);
                }
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        
        if (isLoaded && user) {
            fetchOrders();
        }
    }, [user, isLoaded, router]);

    if (loading) return <div className="text-center py-20">Loading orders...</div>;

    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold mb-8 text-gray-800">Your Orders</h1>
            
            {orders.length === 0 ? (
                <div className="text-center py-12 bg-white rounded-lg shadow">
                    <p className="text-xl text-gray-600 mb-4">No orders found.</p>
                    <Link href="/explore">
                        <button className="bg-redCustom text-white px-6 py-2 rounded-full hover:bg-orangeCustom transition">
                            Start Ordering
                        </button>
                    </Link>
                </div>
            ) : (
                <div className="grid gap-8">
                    {orders.map((order) => (
                        <div key={order._id} className="bg-gradient-to-br from-gray-900 via-gray-800 to-black p-6 rounded-2xl shadow-xl border border-gray-700 text-white relative overflow-hidden transition-all duration-300 hover:scale-[1.01] hover:shadow-2xl hover:shadow-orange-500/10">
                            {/* Decorative Top Bar */}
                            <div className={`absolute top-0 left-0 w-full h-1 ${
                                order.status === 'completed' ? 'bg-gradient-to-r from-green-400 to-emerald-600' : 'bg-gradient-to-r from-yellow-400 to-orange-500'
                            }`} />

                            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 border-b border-gray-700 pb-4 mt-2">
                                <div>
                                    <p className="text-xs text-gray-400 font-mono tracking-wider uppercase mb-1">Order ID</p>
                                    <p className="text-sm font-semibold text-gray-200">{order._id}</p>
                                    <p className="text-xs text-gray-500 mt-1">
                                        {new Date(order.createdAt).toLocaleDateString(undefined, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' })} • {new Date(order.createdAt).toLocaleTimeString()}
                                    </p>
                                </div>
                                <div className="text-right mt-4 md:mt-0 flex flex-col items-end">
                                    <span className={`inline-block px-4 py-1 rounded-full text-xs font-bold tracking-wide shadow-lg mb-2 ${
                                        order.status === 'completed' 
                                        ? 'bg-green-900/50 text-green-400 border border-green-700/50' 
                                        : 'bg-yellow-900/50 text-yellow-400 border border-yellow-700/50'
                                    }`}>
                                        {order.status.toUpperCase()}
                                    </span>
                                    <p className="font-bold text-3xl text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-400">
                                        ₹{order.totalAmount.toFixed(2)}
                                    </p>
                                </div>
                            </div>
                            
                            <div className="space-y-3 mt-4">
                                {order.items.map((item, idx) => (
                                    <div key={idx} className="group flex justify-between items-center bg-white/5 backdrop-blur-sm p-4 rounded-xl border border-white/10 hover:bg-white/10 hover:border-orange-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-orange-500/10">
                                        <div className="flex items-center space-x-5">
                                            <div className="relative">
                                                {item.foodId?.image ? (
                                                    <img src={item.foodId.image} alt={item.foodId.name} className="w-16 h-16 object-cover rounded-xl shadow-md group-hover:scale-105 transition-transform duration-300" />
                                                ) : (
                                                    <div className="w-16 h-16 bg-gray-700 rounded-xl flex items-center justify-center text-gray-500 text-xs">No Img</div>
                                                )}
                                                {/* Quantity Badge */}
                                                <span className="absolute -top-2 -right-2 bg-orange-500 text-white text-xs font-bold w-6 h-6 flex items-center justify-center rounded-full shadow-md border border-gray-900">
                                                    {item.quantity}
                                                </span>
                                            </div>
                                            <div>
                                                <p className="font-bold text-lg text-white group-hover:text-orange-400 transition-colors">{item.foodId?.name || "Unknown Item"}</p>
                                                <p className="text-sm text-gray-400 mt-1">
                                                     ₹{item.price} each
                                                </p>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <p className="font-bold text-xl text-white">₹{(item.quantity * item.price).toFixed(2)}</p>
                                            <p className="text-xs text-gray-500 mt-1">Subtotal</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default OrdersPage;
