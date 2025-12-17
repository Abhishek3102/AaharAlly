"use client";
import React, { useState } from 'react';
import { FaSpinner } from 'react-icons/fa';
import { toast, Toaster } from 'react-hot-toast';
import axios from 'axios';

const ServiceProviderUpload = () => {
    const [file, setFile] = useState(null);
    const [file1, setFile1] = useState(null);
    const [loading, setLoading] = useState(false);
    const [loading1, setLoading1] = useState(false);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleFileChange1 = (event) => {
        setFile1(event.target.files[0]);
    };

    const fetchData = async () => {
        try {
            const response = await fetch('https://2wl9vj7n-5000.inc1.devtunnels.ms/api/get_data', {
                method: "GET",
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.success) {
                console.log(result.data);
                const resp = await axios.post("/api/FoodPreferences/", { ...result.data.data })
                if (!resp.ok) {
                    return
                }
                toast.success("Model 🎉")
                return;
            } else {
                toast.error("Model 😭")
                return;
            }
        } catch (err) {
            toast.error(err.message);
        }
    };

    async function downloadCSV() {
        try {
            toast.loading('Preparing download...', { id: 'download' });

            const response = await fetch('/api/UserData', {
                method: 'GET',
            });

            if (!response.ok) {
                toast.error(response.message)
                throw new Error('Failed to fetch CSV data');
            }

            toast.success("File Given to the ML Model Successfully")
            await fetchData();
            return;
        } catch (error) {
            toast.error(`Error downloading CSV: ${error.message}`, { id: 'download' });
            console.error('Error downloading CSV:', error);
        } finally {
            toast.dismiss('download');
        }
    }


    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) {
            toast.error('Please select a file to upload.');
            return;
        }
        const formData = new FormData();
        formData.append('file', file);
        setLoading(true);

        try {
            const response = await fetch('/api/ServiceProvider', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();
            toast.success(result.message);
        } catch (error) {
            toast.error(`Error uploading file: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit1 = async (event) => {
        event.preventDefault();
        if (!file1) {
            toast.error('Please select a file to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file1);

        setLoading1(true);

        try {
            const response = await fetch('/api/UserData', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();
            toast.success(result.message);
            await downloadCSV();
        } catch (error) {
            toast.error(`Error uploading file: ${error.message}`);
        } finally {
            setLoading1(false);
        }
    };

    return (
        <div className="flex flex-col items-center w-full p-6 bg-gray-50 shadow-lg rounded-lg">
            <Toaster /> {/* Add Toaster component here */}
            <h1 className="text-2xl font-bold text-gray-800 mb-6">Upload Your Data</h1>

            <div className="grid md:grid-cols-2 gap-6 w-full max-w-4xl">
                <div className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
                    <h2 className="text-xl font-semibold mb-4 text-center text-gray-700">Add Your Products</h2>
                    <form onSubmit={handleSubmit} className="flex flex-col items-center">
                        <input
                            type="file"
                            accept=".xlsx, .xls"
                            onChange={handleFileChange}
                            className="mb-4 w-full p-2 border rounded-md text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400"
                        />
                        <button
                            type="submit"
                            className={`flex items-center justify-center py-2 px-4 w-full text-white font-semibold rounded-md transition-colors duration-200 ${loading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'
                                }`}
                            disabled={loading}
                        >
                            {loading ? <FaSpinner className="animate-spin mr-2" /> : null}
                            {loading ? 'Uploading...' : 'Upload'}
                        </button>
                    </form>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
                    <h2 className="text-xl font-semibold mb-4 text-center text-gray-700">Add Users Data</h2>
                    <form onSubmit={handleSubmit1} className="flex flex-col items-center">
                        <input
                            type="file"
                            accept=".xlsx, .xls"
                            onChange={handleFileChange1}
                            className="mb-4 w-full p-2 border rounded-md text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400"
                        />
                        <button
                            type="submit"
                            className={`flex items-center justify-center py-2 px-4 w-full text-white font-semibold rounded-md transition-colors duration-200 ${loading1 ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'
                                }`}
                            disabled={loading1}
                        >
                            {loading1 ? <FaSpinner className="animate-spin mr-2" /> : null}
                            {loading1 ? 'Uploading...' : 'Upload'}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default ServiceProviderUpload;
