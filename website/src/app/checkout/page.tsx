"use client";
import React from 'react';

const CheckoutPage = () => {
  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      <h1 className="text-3xl font-bold mb-8 text-gray-800 text-center">Checkout</h1>
      
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-100">
        <form>
          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">Shipping Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-600 mb-1">First Name</label>
                <input type="text" className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" required />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-600 mb-1">Last Name</label>
                <input type="text" className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" required />
              </div>
            </div>
            <div className="mt-4">
               <label className="block text-sm font-medium text-gray-600 mb-1">Address</label>
               <input type="text" className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" required />
            </div>
             <div className="mt-4 grid grid-cols-2 gap-4">
               <div>
                  <label className="block text-sm font-medium text-gray-600 mb-1">City</label>
                  <input type="text" className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" required />
               </div>
               <div>
                  <label className="block text-sm font-medium text-gray-600 mb-1">Postal Code</label>
                  <input type="text" className="w-full p-2 border border-gray-300 rounded focus:border-redCustom focus:ring-1 focus:ring-redCustom outline-none" required />
               </div>
            </div>
          </div>

          <div className="mb-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">Payment Details</h2>
            <div className="p-4 bg-gray-50 rounded text-center text-gray-500">
                Payment Gateway Integration Coming Soon
            </div>
          </div>

          <button type="submit" className="w-full bg-redCustom text-white font-bold py-3 rounded-lg hover:bg-orangeCustom transition duration-300">
            Place Order
          </button>
        </form>
      </div>
    </div>
  );
};

export default CheckoutPage;
