
"use client";
import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-hot-toast';

const MLControls = () => {
    const [loading, setLoading] = useState("");
    const [result, setResult] = useState<any>(null);

    const handleTrain = async () => {
        setLoading("training");
        setResult(null);
        try {
            const res = await axios.post('/api/ml/train', {});
            setResult(res.data);
            if (res.data.success) toast.success("Model Trained Successfully!");
            else toast.error("Training Failed: " + res.data.error);
        } catch (error: any) {
            toast.error("Error: " + error.message);
        } finally {
            setLoading("");
        }
    };

    const handleUpdate = async () => {
        setLoading("updating");
        setResult(null);
        try {
            const res = await axios.get('/api/cron/refresh');
            setResult(res.data);
            if (res.data.success) toast.success(`Updated Users: ${res.data.message}`);
            else toast.error("Update Failed: " + res.data.message);
        } catch (error: any) {
            toast.error("Error: " + error.message);
        } finally {
            setLoading("");
        }
    };

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 mb-8">
            <h2 className="text-xl font-bold text-gray-800 mb-4">ML Control Center 🧠</h2>
            <div className="flex flex-col md:flex-row gap-4 mb-4">
                <button
                    onClick={handleTrain}
                    disabled={!!loading}
                    className={`px-6 py-3 rounded-lg font-medium text-white transition
                        ${loading === "training" ? "bg-blue-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"}
                    `}
                >
                    {loading === "training" ? "Training Brain..." : "Step 1: Train AI Model"}
                </button>

                <button
                    onClick={handleUpdate}
                    disabled={!!loading}
                    className={`px-6 py-3 rounded-lg font-medium text-white transition
                        ${loading === "updating" ? "bg-purple-400 cursor-not-allowed" : "bg-purple-600 hover:bg-purple-700"}
                    `}
                >
                    {loading === "updating" ? "Generating Recommendations..." : "Step 2: Update Users"}
                </button>
            </div>

            {result && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-100 text-sm overflow-auto max-h-40">
                    <pre>{JSON.stringify(result, null, 2)}</pre>
                </div>
            )}
            
            <p className="text-gray-500 text-sm mt-2">
                * <strong>Train</strong>: Learns from 2000+ synthetic records + Live Orders. <br/>
                * <strong>Update</strong>: Applies new knowledge to all users for better recommendations.
            </p>
        </div>
    );
};

export default MLControls;
