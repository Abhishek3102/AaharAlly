"use client";
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useUser } from "@clerk/nextjs";
import axios from "axios";
import toast from "react-hot-toast";

const CartPage = () => {
    const { user } = useUser();
    const [cartItems, setCartItems] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchCart = async () => {
        if (!user?.primaryEmailAddress?.emailAddress) return;
        try {
            const res = await axios.get(`/api/get-cart?email=${user.primaryEmailAddress.emailAddress}`);
            if (res.data.success) {
                setCartItems(res.data.cart);
            }
        } catch (error) {
            console.error("Error fetching cart", error);
            toast.error("Failed to load cart");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (user) {
            fetchCart();
        } else if (!user && !loading) {
             setLoading(false);
        }
    }, [user]);

    const handleRemove = async (foodId: string) => {
        try {
            const res = await axios.post("/api/remove-from-cart", {
                email: user?.primaryEmailAddress?.emailAddress,
                foodId,
            });
            if (res.data.success) {
                toast.success("Item removed");
                fetchCart(); // Refresh cart
            }
        } catch (error) {
            toast.error("Failed to remove item");
        }
    };

    const calculateTotal = () => {
        return cartItems.reduce((total: number, item: any) => {
             const price = parseFloat(item.foodId?.price?.replace(/[^0-9.]/g, '') || "0");
             return total + (price * item.quantity);
        }, 0);
    };

    if (loading) return <div className="text-center py-20">Loading cart...</div>;

    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold mb-6 text-gray-800">Your Cart</h1>
            
            {cartItems.length === 0 ? (
                <div className="text-center py-20 bg-white rounded-lg shadow-sm">
                    <h2 className="text-xl text-gray-600 mb-4">Your cart is currently empty</h2>
                    <p className="text-gray-500 mb-8">Looks like you haven't added any delicious food yet!</p>
                    <Link href="/" className="bg-redCustom text-white px-6 py-3 rounded-full font-semibold hover:bg-orangeCustom transition duration-300">
                        Start Exploration
                    </Link>
                </div>
            ) : (
                <div className="flex flex-col lg:flex-row gap-8">
                    <div className="flex-1 bg-white p-6 rounded-lg shadow">
                         {cartItems.map((item: any) => (
                             <div key={item._id} className="flex items-center justify-between border-b py-4 last:border-b-0">
                                 <div className="flex items-center gap-4">
                                     <img src={item.foodId?.image} alt={item.foodId?.name} className="w-20 h-20 object-cover rounded-md" />
                                     <div>
                                         <h3 className="font-semibold text-lg">{item.foodId?.name}</h3>
                                         <p className="text-gray-500">${item.foodId?.price}</p>
                                     </div>
                                 </div>
                                 <div className="flex items-center gap-6">
                                     <span className="font-medium">Qty: {item.quantity}</span>
                                     <button 
                                        onClick={() => handleRemove(item.foodId?._id)}
                                        className="text-red-500 hover:text-red-700 font-medium"
                                     >
                                         Remove
                                     </button>
                                 </div>
                             </div>
                         ))}
                    </div>
                    <div className="lg:w-1/3 bg-white p-6 rounded-lg shadow h-fit">
                        <h2 className="text-xl font-bold mb-4">Order Summary</h2>
                        <div className="flex justify-between mb-2">
                            <span>Subtotal</span>
                            <span>${calculateTotal().toFixed(2)}</span>
                        </div>
                        <div className="border-t pt-4 mt-4">
                            <div className="flex justify-between font-bold text-lg">
                                <span>Total</span>
                                <span>${calculateTotal().toFixed(2)}</span>
                            </div>
                            <Link href="/checkout" className="block text-center mt-6 bg-orangeCustom text-white py-3 rounded-lg font-semibold hover:bg-deep-orange-600 transition">
                                Proceed to Checkout
                            </Link>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CartPage;