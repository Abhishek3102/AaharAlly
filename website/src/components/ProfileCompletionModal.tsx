"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";
import toast from "react-hot-toast";
import { useUser } from "@clerk/nextjs";

const ProfileCompletionModal = () => {
  const { isSignedIn, isLoaded } = useUser();
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("male");

  useEffect(() => {
    if (isLoaded && isSignedIn) {
      checkProfile();
    }
  }, [isLoaded, isSignedIn]);

  const checkProfile = async () => {
    try {
      const res = await axios.get("/api/user/profile");
      if (res.data.success) {
        if (!res.data.profileCompleted) {
          setIsOpen(true);
        }
      }
    } catch (error) {
      console.error("Error checking profile", error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!age || !gender) return toast.error("Please fill all fields");

    setLoading(true);
    try {
      const res = await axios.patch("/api/user/profile", {
        age: parseInt(age),
        gender,
      });

      if (res.data.success) {
        toast.success("Profile Updated!");
        setIsOpen(false);
      }
    } catch (error) {
      toast.error("Failed to update profile");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm">
      <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md border border-gray-100">
        <h2 className="text-3xl font-bold mb-2 text-gray-800 text-center">One Last Step! 🚀</h2>
        <p className="text-gray-500 mb-6 text-center text-sm">
          To give you the best food recommendations, we simply need a few details.
        </p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-5">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-semibold text-gray-700">Age</label>
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="border border-gray-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-orangeCustom/50 transition-all font-medium"
              placeholder="e.g. 25"
              min="1"
              max="120"
              required
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-semibold text-gray-700">Gender</label>
            <div className="flex gap-4">
               {['Male', 'Female', 'Other'].map((option) => (
                 <label 
                    key={option} 
                    className={`flex-1 cursor-pointer border rounded-lg p-3 text-center transition-all ${
                        gender.toLowerCase() === option.toLowerCase() 
                        ? 'bg-orangeCustom text-white border-orangeCustom shadow-md' 
                        : 'bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100'
                    }`}
                 >
                    <input 
                        type="radio" 
                        name="gender" 
                        value={option.toLowerCase()}
                        checked={gender.toLowerCase() === option.toLowerCase()}
                        onChange={(e) => setGender(e.target.value)}
                        className="hidden"
                    />
                    <span className="font-medium">{option}</span>
                 </label>
               ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="mt-4 bg-orangeCustom text-white py-3 rounded-lg font-bold text-lg hover:shadow-lg hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {loading ? "Saving..." : "Start Exploring"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ProfileCompletionModal;
