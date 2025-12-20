"use client";
import React from 'react';
import axios from 'axios';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';

const CheckoutPage = () => {
  const router = useRouter();
  const [loading, setLoading] = React.useState(false);
  const [formData, setFormData] = React.useState({
      firstName: '',
      lastName: '',
      address: '',
      city: '',
      postalCode: ''
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
        const res = await axios.post('/api/orders', {
            shippingInfo: formData
        });

        if (res.data.success) {
            toast.success("Order placed successfully!");
            router.push('/orders');
        } else {
            toast.error(res.data.message || "Failed to place order");
        }
    } catch (error: any) {
        console.error("Checkout Error:", error);
        toast.error(error.response?.data?.message || "Something went wrong");
    } finally {
        setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      <h1 className="text-3xl font-bold mb-8 text-gray-800 text-center">Checkout</h1>
      
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-100">
        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">Shipping Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-600 mb-1">First Name</label>
                <input 
                    type="text" 
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                    required 
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-600 mb-1">Last Name</label>
                <input 
                    type="text" 
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                    required 
                />
              </div>
            </div>
            <div className="mt-4">
               <label className="block text-sm font-medium text-gray-600 mb-1">Address</label>
               <input 
                    type="text" 
                    name="address"
                    value={formData.address}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                    required 
                />
            </div>
             <div className="mt-4 grid grid-cols-2 gap-4">
               <div>
                  <label className="block text-sm font-medium text-gray-600 mb-1">City</label>
                  <input 
                    type="text" 
                    name="city"
                    value={formData.city}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                    required 
                   />
               </div>
               <div>
                  <label className="block text-sm font-medium text-gray-600 mb-1">Postal Code</label>
                  <input 
                    type="text" 
                    name="postalCode"
                    value={formData.postalCode}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" 
                    required 
                   />
               </div>
            </div>
          </div>

          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">Payment Details</h2>
            <div className="p-4 bg-gray-50 rounded text-center text-gray-500">
                Payment Gateway Integration Coming Soon (Cash on Delivery)
            </div>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full bg-redCustom text-white font-bold py-3 rounded-lg hover:bg-orangeCustom transition duration-300 disabled:opacity-70"
          >
            {loading ? "Placing Order..." : "Place Order"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default CheckoutPage;
