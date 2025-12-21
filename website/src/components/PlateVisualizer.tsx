// @ts-nocheck
"use client";

import React, { useState } from "react";
import { Button, Typography, Switch, Card, CardBody } from "@material-tailwind/react";
import { FaMagic, FaUndo } from "react-icons/fa";
import Image from "next/image";
import toast from "react-hot-toast";

interface PlateVisualizerProps {
  foodName: string;
  baseImage: string;
}

const MODIFIERS = [
  "Extra Cheese 🧀",
  "Crispy Bacon 🥓",
  "Spicy Jalapenos 🌶️",
  "Fresh Avocado 🥑",
  "Double Patty 🍔",
  "No Bun (Lettuce Wrap) 🥬",
  "Onion Rings on top 🧅",
  "Fried Egg 🍳"
];

const PlateVisualizer: React.FC<PlateVisualizerProps> = ({ foodName, baseImage }) => {
  const [selectedMods, setSelectedMods] = useState<string[]>([]);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const toggleMod = (mod: string) => {
    setSelectedMods(prev => 
      prev.includes(mod) ? prev.filter(m => m !== mod) : [...prev, mod]
    );
  };

  const handleVisualize = async () => {
    if (selectedMods.length === 0) {
        toast.error("Select at least one customization!");
        return;
    }

    setLoading(true);
    setGeneratedImage(null); // Clear previous

    try {
      const res = await fetch("/api/visualize-food", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          foodName,
          modifiers: selectedMods
        })
      });

      const data = await res.json();

      if (data.success && data.imageUrl) {
        setGeneratedImage(data.imageUrl);
        toast.success("Plate visualized!");
      } else {
        toast.error("Failed to generate image. Try again.");
        console.error(data.error);
      }
    } catch (err) {
      console.error(err);
      toast.error("Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full mt-6 border border-gray-200 shadow-sm" placeholder={undefined}>
      <CardBody placeholder={undefined}>
        <div className="flex flex-col md:flex-row gap-8">
            
          {/* LEFT: Customization Panel */}
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-4">
                <FaMagic className="text-purple-600" />
                <Typography variant="h5" color="blue-gray" className="font-bold" placeholder={undefined}>
                AI Plate Customizer
                </Typography>
            </div>
            <Typography variant="small" className="mb-4 text-gray-500" placeholder={undefined}>
                Select ingredients to see a <b>Real-time AI Generation</b> of your custom plate.
            </Typography>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
                {MODIFIERS.map((mod) => (
                    <div key={mod} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                        <span className="text-sm font-medium text-gray-700">{mod}</span>
                        <Switch 
                            crossOrigin={undefined}
                            onChange={() => toggleMod(mod)}
                            checked={selectedMods.includes(mod)}
                            color="purple"
                        />
                    </div>
                ))}
            </div>

            <Button 
                fullWidth 
                color="purple" 
                onClick={handleVisualize}
                disabled={loading}
                className="flex justify-center items-center gap-2"
                placeholder={undefined}
            >
                {loading ? "Chef is cooking... (5s)" : "✨ Visualize My Plate"}
            </Button>
          </div>

          {/* RIGHT: Image Display */}
          <div className="flex-1 flex flex-col items-center justify-center relative min-h-[300px] bg-gray-100 rounded-xl overflow-hidden border-2 border-dashed border-gray-300">
            {/* Base or Generated Image */}
            <div className="relative w-full h-full min-h-[300px]">
                <Image 
                    src={generatedImage || baseImage}
                    alt="Food Preview"
                    fill
                    className={`object-cover transition-opacity duration-500 ${loading ? "opacity-50 blur-sm" : "opacity-100"}`}
                />
                
                {/* Overlay for Loading */}
                {loading && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/30 z-10">
                         <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white mb-2"></div>
                         <p className="text-white font-bold animate-pulse">Generating 3D View...</p>
                    </div>
                )}

                {/* Badge */}
                {generatedImage && !loading && (
                    <div className="absolute top-4 right-4 bg-purple-600 text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg flex items-center gap-1">
                        ✨ AI Generated
                    </div>
                )}
            </div>
            
            {generatedImage && (
                <button 
                    onClick={() => setGeneratedImage(null)}
                    className="absolute bottom-4 left-4 bg-white/90 p-2 rounded-full shadow hover:bg-white text-gray-700 text-xs flex items-center gap-1 z-20"
                >
                    <FaUndo /> Reset
                </button>
            )}
          </div>

        </div>
      </CardBody>
    </Card>
  );
};

export default PlateVisualizer;
