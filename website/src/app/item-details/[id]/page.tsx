"use client";
import Image from "next/image";
import React, { useEffect, useState } from "react";
import { Food } from "@/types";
import axios from "axios";
import Loading from "@/components/loading";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { useUser } from "@clerk/nextjs";
import toast from "react-hot-toast";
import PlateVisualizer from "@/components/PlateVisualizer";


const IngredientsModal = ({ ingredients, onClose }: { ingredients: string[]; onClose: () => void }) => (
  <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
    <div className="bg-white p-5 rounded-lg shadow-lg max-w-md w-full">
      <h2 className="text-xl font-semibold mb-4">Ingredients</h2>
      <ul className="list-disc pl-5 text-gray-600">
        {ingredients.map((ingredient, index) => (
          <li key={index}>{ingredient}</li>
        ))}
      </ul>
      <button onClick={onClose} className="mt-4 bg-red-500 text-white px-4 py-2 rounded">
        Close
      </button>
    </div>
  </div>
);

function convertPrice(priceString: string) {
  // Handle ranges and clean currency symbols
  let p = priceString || "0";
  if (p.includes('-')) {
      p = p.split('-')[0];
  }
  return parseFloat(p.replace(/[^0-9.]/g, '') || "0");
}

const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GENAI!);
const TacoCard = ({ params }: { params: Promise<{ id: string }> }) => {
  const [loading, setLoading] = useState(false);
  const [itemDetails, setItemDetails] = useState<Food>();
  const [ingredients, setIngredients] = useState<string[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [id, setId] = useState<string | null>(null);
  const { user } = useUser();

  // Dynamic Pairings State
  const [pairings, setPairings] = useState<Food[]>([]);
  const [selectedPairings, setSelectedPairings] = useState<Food[]>([]);

  const addToCart = async () => {
    if (!user) {
      toast.error("Please sign in to add items to cart");
      return;
    }
    if (!itemDetails) return;

    try {
      toast.loading("Adding items to cart...");
      
      // 1. Add Main Item
      await axios.post("/api/add-to-cart", {
        email: user.primaryEmailAddress?.emailAddress,
        foodId: itemDetails._id,
      });

      // 2. Add Selected Pairings
      for (const pairing of selectedPairings) {
         await axios.post("/api/add-to-cart", {
            email: user.primaryEmailAddress?.emailAddress,
            foodId: pairing._id,
         });
      }

      toast.dismiss();
      toast.success(`Added ${1 + selectedPairings.length} items to cart!`);
    } catch (error) {
      toast.dismiss();
      toast.error("Failed to add to cart");
      console.error(error);
    }
  };

  useEffect(() => {
    const fetchParams = async () => {
      const resolvedParams = await params; // Unwrap the Promise
      setId(resolvedParams.id);
    };
    fetchParams();
  }, [params]);

  // Fetch Item + Dynamic Pairings
  useEffect(() => {
    const controller = new AbortController();
    const fetchDetails = async () => {
      try {
        setLoading(true);
        // 1. Fetch Main Item
        const resp = await axios.get("/api/Users/", {
          params: { id },
          signal: controller.signal,
        });
        const mainItem = resp.data.data;
        setItemDetails(mainItem);

        // 2. Fetch Pairings (Random "Snacks" or "Beverages")
        // We fetch a broader list and pick 3 random ones that are NOT the main item
        const pairingsResp = await axios.get("/api/Users", {
           params: { categories: "Snacks,Beverages,Street Food" },
           signal: controller.signal 
        });
        
        if (pairingsResp.data.success) {
            const allCandidates = pairingsResp.data.data;
            // Filter out current item and pick 3 random
            const filtered = allCandidates.filter((f: Food) => f._id !== mainItem._id);
            const shuffled = filtered.sort(() => 0.5 - Math.random());
            setPairings(shuffled.slice(0, 3));
        }

      } catch (error) {
        console.log(error);
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchDetails();
    }
    return () => controller.abort();
  }, [id]);

  const togglePairing = (food: Food) => {
      if (selectedPairings.find(p => p._id === food._id)) {
          setSelectedPairings(selectedPairings.filter(p => p._id !== food._id));
      } else {
          setSelectedPairings([...selectedPairings, food]);
      }
  };

  const getTotalPrice = () => {
      if (!itemDetails) return 0;
      let total = convertPrice(itemDetails.price);
      selectedPairings.forEach(p => {
          total += convertPrice(p.price);
      });
      return total;
  };

  async function fetchIngredients(item: string) {
    try {
      const model = genAI.getGenerativeModel({
        model: "gemini-flash-lite-latest",
      });
      const prompt = `List the ingredients for ${item} as a JSON array of only 7 items.`;

      // Pass the prompt directly as a string to generateContent
      const result = await model.generateContent(prompt);
      const responseText = await result.response.text();

      console.log("Response from model:", responseText); // Inspect the raw response

      // Attempt to find JSON array of ingredients in the response
      const jsonMatch = responseText.match(/\[([\s\S]*?)\]/);
      if (jsonMatch) {
        const parsedIngredients = JSON.parse(jsonMatch[0]);
        setIngredients(parsedIngredients);
        setShowModal(true); // Show modal after setting ingredients
      } else {
        console.warn("No JSON array found in response");
        setIngredients(["Ingredient data not found."]);
      }
    } catch (error) {
      console.error("Error fetching ingredients:", error);
      setIngredients(["Error fetching ingredients."]);
      setShowModal(true); // Show modal even on error
    }
  }

  return (
    <>
      {loading && <Loading />}
      {showModal && (
        <IngredientsModal
          ingredients={ingredients}
          onClose={() => setShowModal(false)}
        />
      )}
      {!loading && itemDetails && (
        <div className="w-full max-w-full min-h-full my-auto mx-auto bg-white rounded-xl p-5 lg:p-8 transition-all duration-300 flex flex-col lg:flex-row">
          <div className="relative w-full lg:w-1/3 h-full lg:h-auto mb-5 lg:mb-0 lg:mr-8">
            <Image
              src={itemDetails.image}
              alt={itemDetails.name}
              width={400}
              height={400}
              className="rounded-xl w-full h-[75vh] object-cover shadow-lg shadow-gray-600 hover:scale-105 hover:shadow-xl hover:shadow-gray-500 transition duration-500 transform ease-in-out"
            />
            <button className="absolute top-2 right-2 text-2xl text-red-500 focus:outline-none">
              ❤️
            </button>
          </div>

          <div className="flex-grow flex flex-col justify-between">
            <div>
              <h2 className="text-xl md:text-2xl lg:text-4xl font-semibold">
                {itemDetails.name}
              </h2>
              <div className="flex justify-start items-center text-gray-600 mt-2">
                <span className="text-sm md:text-base mx-4">
                  ⭐ {itemDetails.rating}
                </span>
                <div className="flex flex-col md:flex-row justify-start items-center mt-2">
                  <div className="flex justify-start items-center">
                    <p className="text-2xl md:text-3xl text-orange-500 font-bold">
                      $ {getTotalPrice().toFixed(2)}
                    </p>
                    {selectedPairings.length > 0 && (
                        <span className="text-sm text-gray-400 ml-2">
                            (Base: ${convertPrice(itemDetails.price)} + Pairings)
                        </span>
                    )}
                  </div>
                </div>
              </div>

              <p className="text-gray-700 mt-3 text-sm md:text-lg">
                {itemDetails.description}
              </p>

              <div className="flex flex-col justify-start items-start mt-8">
                <span className="font-semibold text-lg mb-2">Complete the Meal (Pair me with):</span>
                <div className="flex flex-wrap gap-3">
                  {pairings.length > 0 ? pairings.map((pairing) => {
                    const isSelected = selectedPairings.some(p => p._id === pairing._id);
                    return (
                        <div 
                            key={pairing._id}
                            onClick={() => togglePairing(pairing)}
                            className={`cursor-pointer border rounded-lg p-2 flex items-center gap-2 transition-all duration-200 select-none ${
                                isSelected 
                                ? 'bg-orange-50 border-orange-500 shadow-md ring-1 ring-orange-500' 
                                : 'bg-white border-gray-200 hover:border-orange-300 hover:shadow-sm'
                            }`}
                        >
                            <img src={pairing.image} alt={pairing.name} className="w-10 h-10 rounded object-cover" />
                            <div className="flex flex-col">
                                <span className={`text-sm font-medium ${isSelected ? 'text-orange-700' : 'text-gray-700'}`}>
                                    {pairing.name}
                                </span>
                                <span className="text-xs text-gray-500">
                                    + ${convertPrice(pairing.price)}
                                </span>
                            </div>
                            {isSelected && <span className="text-orange-500 ml-1">✓</span>}
                        </div>
                    );
                  }) : (
                      <span className="text-sm text-gray-400">No suggestions available.</span>
                  )}
                </div>
              </div>

              {/* Nutritional Info */}
              <div className="mt-8 border-t pt-4">
                <h3 className="text-lg font-semibold">Nutritional Info:</h3>
                <ul className="list-disc pl-5 text-gray-600">
                  <li>Calories: 250</li>
                  <li>Protein: 12g</li>
                  <li>Carbs: 30g</li>
                  <li>Fat: 10g</li>
                </ul>
              </div>

              {/* User Reviews */}
              <div className="mt-5 border-t pt-4">
                <h3 className="text-lg font-semibold">User Reviews:</h3>
                <div className="text-gray-600">
                  <p><strong>Jane:</strong> Absolutely delicious&#x21;</p>
                  <p><strong>Mark:</strong> The best thing Ive ever had&#x21;</p>
                </div>
              </div>
            </div>
            
            {/* GenAI Plate Visualizer (COMING SOON) */}
            <div className="mb-8 p-6 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-xl border border-purple-100 flex flex-col items-center text-center">
                <div className="bg-white p-3 rounded-full shadow-sm mb-3">
                    <span className="text-2xl">🎨</span>
                </div>
                <h3 className="text-lg font-bold text-gray-800">AI Plate Customizer (Coming Soon)</h3>
                <p className="text-sm text-gray-600 max-w-md mt-2">
                    Soon you'll be able to visually customize your dish (Add Cheese, Remove Bun) and see a 
                    <span className="font-semibold text-purple-600"> Real-time 3D Generative Preview</span> before you order!
                </p>
                <div className="mt-4 text-xs font-semibold text-purple-500 bg-purple-100 px-3 py-1 rounded-full">
                    🚧 Under Construction
                </div>
            </div>

            <div className="flex justify-between items-center mt-5 mb-10">
              <button
                onClick={addToCart}
                className="bg-orangeCustom text-white px-6 py-3 rounded-lg shadow-lg shadow-orangeCustom hover:bg-deep-orange-600 hover:shadow-deep-orange-700 transition duration-500 font-bold text-lg"
              >
                Add {1 + selectedPairings.length} items to Cart
              </button>
              <button
                onClick={() => fetchIngredients(itemDetails.name)}
                className="bg-greenCustom text-white px-4 py-2 rounded-lg shadow-lg shadow-greenCustom hover:bg-light-green-700 hover:shadow-light-green-700 transition duration-500"
              >
                Ingredients
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TacoCard;
