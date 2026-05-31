"use client";
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Link from 'next/link';
import { useUser } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { toast, Toaster } from 'react-hot-toast';

interface OrderItem {
    foodId: {
        _id: string;
        name: string;
        image: string;
        price: string;
    } | any;
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

const OrdersPage = () => {
    const { user, isLoaded } = useUser();
    const router = useRouter();
    const [orders, setOrders] = useState<Order[]>([]);
    const [loading, setLoading] = useState(true);
    
    // Review State
    const [reviewingId, setReviewingId] = useState<string | null>(null);
    const [reviewText, setReviewText] = useState("");
    const [submittingReview, setSubmittingReview] = useState(false);

    useEffect(() => {
        if (isLoaded && !user) {
            router.push('/sign-in'); 
            return;
        }

        const fetchOrders = async () => {
            if (!user) return;
            try {
                const res = await axios.get('/api/orders');
                if (res.data.success) {
                    setOrders(res.data.orders);
                }
            } catch (err) {
                console.error("Fetch Orders Error:", err);
            } finally {
                setLoading(false);
            }
        };
        
        if (isLoaded && user) {
            fetchOrders();
        }
    }, [user, isLoaded, router]);

    const handleReviewSubmit = async (foodId: string) => {
        if (!reviewText.trim()) {
            toast.error("Please enter a review");
            return;
        }

        setSubmittingReview(true);
        try {
            const res = await axios.post('/api/orders/review', {
                foodId,
                reviewText
            });

            if (res.data.success) {
                toast.success(res.data.message, { icon: res.data.isPositive ? '🔥' : '❄️' });
                setReviewingId(null);
                setReviewText("");
            }
        } catch (err: any) {
            toast.error(err.response?.data?.message || "Failed to submit review");
        } finally {
            setSubmittingReview(false);
        }
    };

    if (loading) return <div className="text-center py-20 text-white bg-black min-h-screen">Loading orders...</div>;

    return (
        <div className="bg-black min-h-screen pb-20">
            <Toaster position="bottom-center" />
            <div className="container mx-auto px-4 py-8">
                <h1 className="text-3xl font-bold mb-8 text-white text-center md:text-left">Your Order History</h1>
                
                {orders.length === 0 ? (
                    <div className="text-center py-12 bg-white/5 backdrop-blur-md rounded-lg shadow border border-white/10">
                        <p className="text-xl text-gray-400 mb-4">No orders found.</p>
                        <Link href="/explore">
                            <button className="bg-orange-500 text-white px-6 py-2 rounded-full hover:bg-orange-600 transition">
                                Explore Menu
                            </button>
                        </Link>
                    </div>
                ) : (
                    <div className="grid gap-12">
                        {orders.map((order) => (
                            <div key={order._id} className="bg-gradient-to-br from-gray-900 via-gray-800 to-black p-6 rounded-3xl shadow-2xl border border-gray-800 text-white relative overflow-hidden transition-all duration-300 hover:shadow-orange-500/10">
                                {/* Status Header */}
                                <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-10 border-b border-white/10 pb-6">
                                    <div className="flex flex-col">
                                        <div className="flex items-center gap-3">
                                            <span className={`w-3 h-3 rounded-full animate-pulse ${order.status === 'completed' ? 'bg-green-500' : 'bg-yellow-500'}`} />
                                            <h2 className="text-xs font-mono text-gray-400 uppercase tracking-widest">Order Reference</h2>
                                        </div>
                                        <p className="text-sm font-bold text-gray-300 mt-1">{order._id}</p>
                                        <p className="text-xs text-gray-500 mt-2">
                                            {new Date(order.createdAt).toDateString()} at {new Date(order.createdAt).toLocaleTimeString()}
                                        </p>
                                    </div>
                                    <div className="mt-4 md:mt-0 text-right">
                                        <p className="text-3xl font-black text-white">₹{order.totalAmount.toFixed(2)}</p>
                                        <span className="text-[10px] text-orange-400 font-bold uppercase tracking-tighter">Verified Order</span>
                                    </div>
                                </div>
                                
                                <div className="space-y-6">
                                    {order.items.map((item, idx) => (
                                        <div key={idx} className="flex flex-col bg-white/5 rounded-2xl border border-white/5 overflow-hidden group hover:border-orange-500/20 transition-all duration-500">
                                            <div className="p-5 flex justify-between items-center sm:items-start flex-wrap gap-4">
                                                <div className="flex items-center space-x-6">
                                                    <div className="relative">
                                                        <img 
                                                            src={item.foodId?.image || "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=100"} 
                                                            alt="" 
                                                            className="w-20 h-20 object-cover rounded-2xl shadow-xl group-hover:scale-110 transition-transform duration-500" 
                                                        />
                                                        <div className="absolute -top-3 -right-3 bg-redCustom text-white text-[10px] font-black w-7 h-7 flex items-center justify-center rounded-full border-2 border-black">
                                                            {item.quantity}x
                                                        </div>
                                                    </div>
                                                    <div>
                                                        <h3 className="font-bold text-xl text-white group-hover:text-orange-400 transition-colors">
                                                            {item.foodId?.name || "Premium Meal Item"}
                                                        </h3>
                                                        <p className="text-xs text-gray-500 uppercase font-bold tracking-widest mt-1">
                                                            Ref: {item.foodId?._id ? item.foodId._id.substring(0,8) : 'N/A'}
                                                        </p>
                                                        <p className="text-sm text-gray-400 mt-2 font-mono italic">Unit Price: ₹{item.price}</p>
                                                    </div>
                                                </div>

                                                <div className="flex flex-col items-end gap-3 w-full sm:w-auto">
                                                    <p className="text-2xl font-bold text-white">₹{(item.quantity * item.price).toFixed(2)}</p>
                                                    <button 
                                                        onClick={() => {
                                                            const id = item.foodId?._id || item.foodId;
                                                            if (!id) return;
                                                            setReviewingId(reviewingId === id ? null : id);
                                                        }}
                                                        className="text-[10px] font-black tracking-widest text-orange-400 border border-orange-400/20 px-6 py-2 rounded-full hover:bg-orange-400 hover:text-black transition-all"
                                                    >
                                                        {reviewingId === (item.foodId?._id || item.foodId) ? "ABORT REVIEW" : "RATE & REVIEW ITEM"}
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Expandable Review Panel */}
                                            {reviewingId === (item.foodId?._id || item.foodId) && (
                                                <div className="px-5 py-5 bg-orange-500/5 border-t border-white/10 animate-in fade-in slide-in-from-top-2 duration-300">
                                                    <textarea 
                                                        value={reviewText}
                                                        onChange={(e) => setReviewText(e.target.value)}
                                                        className="w-full bg-black/60 border border-white/10 rounded-2xl p-4 text-sm text-white placeholder:text-gray-600 focus:outline-none focus:border-orange-500/40 min-h-[100px]"
                                                        placeholder="Write your honest review here... Your feedback directly trains our recommendation engine."
                                                    />
                                                    <div className="flex justify-end mt-4">
                                                        <button 
                                                            disabled={submittingReview}
                                                            onClick={() => handleReviewSubmit(item.foodId?._id || item.foodId)}
                                                            className="bg-orange-500 text-black text-[11px] font-black px-8 py-3 rounded-full hover:bg-orange-400 disabled:bg-gray-800 disabled:text-gray-500 transition-all shadow-xl shadow-orange-500/10"
                                                        >
                                                            {submittingReview ? "ANALYZING SENTIMENT..." : "CAST REVIEW"}
                                                        </button>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default OrdersPage;
