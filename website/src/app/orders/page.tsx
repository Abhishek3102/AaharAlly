
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

const OrdersPage = () => {
    const [orders, setOrders] = useState<Order[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchOrders = async () => {
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
        fetchOrders();
    }, []);

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
                <div className="space-y-6">
                    {orders.map((order) => (
                        <div key={order._id} className="bg-white p-6 rounded-lg shadow-md border-l-4 border-redCustom">
                            <div className="flex justify-between items-start mb-4 border-b pb-2">
                                <div>
                                    <p className="text-sm text-gray-500">Order ID: {order._id}</p>
                                    <p className="text-sm text-gray-500">{new Date(order.createdAt).toLocaleDateString()} at {new Date(order.createdAt).toLocaleTimeString()}</p>
                                </div>
                                <div className="text-right">
                                    <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                                        order.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                                    }`}>
                                        {order.status.toUpperCase()}
                                    </span>
                                    <p className="font-bold text-xl mt-1">₹{order.totalAmount.toFixed(2)}</p>
                                </div>
                            </div>
                            
                            <div className="space-y-3">
                                {order.items.map((item, idx) => (
                                    <div key={idx} className="flex justify-between items-center">
                                        <div className="flex items-center space-x-4">
                                            {item.foodId?.image && (
                                                <img src={item.foodId.image} alt={item.foodId.name} className="w-12 h-12 object-cover rounded" />
                                            )}
                                            <div>
                                                <p className="font-medium text-gray-800">{item.foodId?.name || "Unknown Item"}</p>
                                                <p className="text-sm text-gray-500">Qty: {item.quantity} x ₹{item.price}</p>
                                            </div>
                                        </div>
                                        <p className="font-medium">₹{(item.quantity * item.price).toFixed(2)}</p>
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
