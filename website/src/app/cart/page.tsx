"use client";
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useUser } from "@clerk/nextjs";
import axios from "axios";
import toast from "react-hot-toast";
import { useRouter } from "next/navigation";

const CartPage = () => {
    const { user } = useUser();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [cartItems, setCartItems] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    
    // Checkout Modal State
    const [showCheckout, setShowCheckout] = useState(false);
    const [placingOrder, setPlacingOrder] = useState(false);
    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        address: '',
        city: '',
        postalCode: ''
    });
    const router = useRouter(); // Import needed

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handlePlaceOrder = async (e: React.FormEvent) => {
        e.preventDefault();
        setPlacingOrder(true);
        try {
            const res = await axios.post('/api/orders', {
                shippingInfo: formData
            });

            if (res.data.success) {
                toast.success("Order placed successfully!");
                setShowCheckout(false);
                setCartItems([]); // Clear local cart immediately
                router.push('/orders');
            } else {
                toast.error(res.data.message || "Failed to place order");
            }
        } catch (error: any) {
            console.error("Checkout Error:", error);
            toast.error(error.response?.data?.message || "Something went wrong");
        } finally {
            setPlacingOrder(false);
        }
    };

    const fetchCart = React.useCallback(async () => {
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
    }, [user]);

    useEffect(() => {
        if (user) {
            fetchCart();
        } else if (!user && !loading) {
             setLoading(false);
        }
    }, [user, loading, fetchCart]);

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
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } catch (error: any) {
            console.error(error);
            toast.error("Failed to remove item");
        }
    };

    const calculateTotal = () => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return cartItems.reduce((total: number, item: any) => {
             // Fix: Handle ranges like "$12-16" by taking the first part
             let priceStr = item.foodId?.price || "0";
             // If range, take min
             if (priceStr.includes('-')) {
                 priceStr = priceStr.split('-')[0];
             }
             // Clean non-numeric except dot
             const price = parseFloat(priceStr.replace(/[^0-9.]/g, '') || "0");
             return total + (price * item.quantity);
        }, 0);
    };

    // Checkout Modal Component
    const CheckoutModal = () => (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex justify-center items-center p-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
                <div className="p-6">
                    <div className="flex justify-between items-center mb-6">
                        <h2 className="text-2xl font-bold text-gray-800">Checkout</h2>
                        <button onClick={() => setShowCheckout(false)} className="text-gray-500 hover:text-gray-700">
                            ✕
                        </button>
                    </div>

                    <form onSubmit={handlePlaceOrder}>
                        <div className="mb-6">
                            <h3 className="text-lg font-semibold mb-4 text-gray-700">Shipping Information</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-600 mb-1">First Name</label>
                                    <input 
                                        type="text" name="firstName" value={formData.firstName} onChange={handleChange}
                                        className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                                        required 
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-600 mb-1">Last Name</label>
                                    <input 
                                        type="text" name="lastName" value={formData.lastName} onChange={handleChange}
                                        className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                                        required 
                                    />
                                </div>
                            </div>
                            <div className="mt-4">
                               <label className="block text-sm font-medium text-gray-600 mb-1">Address</label>
                               <input 
                                    type="text" name="address" value={formData.address} onChange={handleChange}
                                    className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                                    required 
                                />
                            </div>
                             <div className="mt-4 grid grid-cols-2 gap-4">
                               <div>
                                  <label className="block text-sm font-medium text-gray-600 mb-1">City</label>
                                  <input 
                                    type="text" name="city" value={formData.city} onChange={handleChange}
                                    className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                                    required 
                                   />
                               </div>
                               <div>
                                  <label className="block text-sm font-medium text-gray-600 mb-1">Postal Code</label>
                                  <input 
                                    type="text" name="postalCode" value={formData.postalCode} onChange={handleChange}
                                    className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                                    required 
                                   />
                               </div>
                            </div>
                        </div>

                        <div className="mb-6">
                            <h3 className="text-lg font-semibold mb-4 text-gray-700">Payment Details</h3>
                            <div className="p-4 bg-gray-50 rounded text-center text-gray-500 text-sm">
                                Cash on Delivery (Standard)
                            </div>
                        </div>

                        <div className="flex gap-4">
                            <button 
                                type="button" 
                                onClick={() => setShowCheckout(false)}
                                className="w-1/2 bg-gray-200 text-gray-800 font-semibold py-3 rounded-lg hover:bg-gray-300 transition"
                            >
                                Cancel
                            </button>
                            <button 
                                type="submit" 
                                disabled={placingOrder}
                                className="w-1/2 bg-greenCustom text-white font-bold py-3 rounded-lg hover:bg-green-700 transition disabled:opacity-70"
                            >
                                {placingOrder ? "Processing..." : `Pay $${calculateTotal().toFixed(2)}`}
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    );

    if (loading) return <div className="text-center py-20">Loading cart...</div>;

    return (
        <div className="container mx-auto px-4 py-8 relative">
            <h1 className="text-3xl font-bold mb-6 text-gray-800">Your Cart</h1>
            
            {showCheckout && <CheckoutModal />}

            {cartItems.length === 0 ? (
                <div className="text-center py-20 bg-white rounded-lg shadow-sm">
                    <h2 className="text-xl text-gray-600 mb-4">Your cart is currently empty</h2>
                    <p className="text-gray-500 mb-8">Looks like you haven&apos;t added any delicious food yet!</p>
                    <Link href="/" className="bg-redCustom text-white px-6 py-3 rounded-full font-semibold hover:bg-orangeCustom transition duration-300">
                        Start Exploration
                    </Link>
                </div>
            ) : (
                <div className="flex flex-col lg:flex-row gap-8">
                    <div className="flex-1 bg-white p-6 rounded-lg shadow">
                         {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                         {cartItems.map((item: any) => (
                             <div key={item._id} className="flex items-center justify-between border-b py-4 last:border-b-0">
                                 <div className="flex items-center gap-4">
                                     {/* eslint-disable-next-line @next/next/no-img-element */}
                                     <img src={item.foodId?.image} alt={item.foodId?.name} className="w-20 h-20 object-cover rounded-md" />
                                     <div>
                                         <h3 className="font-semibold text-lg">{item.foodId?.name}</h3>
                                         <p className="text-gray-500">{item.foodId?.price}</p>
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
                            <button 
                                onClick={() => setShowCheckout(true)}
                                className="block w-full text-center mt-6 bg-orangeCustom text-white py-3 rounded-lg font-semibold hover:bg-deep-orange-600 transition"
                            >
                                Buy Now
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CartPage;