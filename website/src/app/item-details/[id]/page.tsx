"use client";
import Image from "next/image";
import React, { useEffect, useState } from "react";
import { Food } from "@/types";
import axios from "axios";
import Loading from "@/components/loading";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { useUser } from "@clerk/nextjs";
import toast from "react-hot-toast";


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
  const priceRange = priceString.split("-");
  const minPrice = parseFloat(priceRange[0].slice(1));
  return minPrice;
}

const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GENAI!);
const TacoCard = ({ params }: { params: Promise<{ id: string }> }) => {
  const [loading, setLoading] = useState(false);
  const [itemDetails, setItemDetails] = useState<Food>();
  const [ingredients, setIngredients] = useState<string[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [id, setId] = useState<string | null>(null);
  const { user } = useUser();

  const addToCart = async () => {
    if (!user) {
      toast.error("Please sign in to add items to cart");
      return;
    }
    if (!itemDetails) return;

    try {
      toast.loading("Adding to cart...");
      const res = await axios.post("/api/add-to-cart", {
        email: user.primaryEmailAddress?.emailAddress,
        foodId: itemDetails._id,
      });
      toast.dismiss();
      if (res.data.success) {
        toast.success("Added to cart!");
      }
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

  const pairings = ["Mojito", "French Fries", "Chips"];

  useEffect(() => {
    const controller = new AbortController();
    const fetchDetails = async () => {
      try {
        setLoading(true);
        const resp = await axios.get("/api/Users/", {
          params: { id },
          signal: controller.signal,
        });
        setItemDetails(resp.data.data);
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
                      $ {convertPrice(itemDetails.price)}
                    </p>
                  </div>
                </div>
              </div>

              <p className="text-gray-700 mt-3 text-sm md:text-lg">
                {itemDetails.description}
              </p>

              <div className="flex justify-start items-center mt-5">
                <span className="font-semibold">Pair me with:</span>
                <div className="flex flex-wrap gap-2 ml-2">
                  {pairings.map((pairing, index) => (
                    <span
                      key={index}
                      className="bg-gray-200 text-gray-600 p-1 rounded"
                    >
                      {pairing}
                    </span>
                  ))}
                </div>
              </div>

              {/* Nutritional Info */}
              <div className="mt-5 border-t pt-4">
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

            <div className="flex justify-between items-center mt-5">
              <button
                onClick={addToCart}
                className="bg-orangeCustom text-white p-2 rounded-lg shadow-lg shadow-orangeCustom hover:bg-deep-orange-600 hover:shadow-deep-orange-700 transition duration-500"
              >
                Add to Cart
              </button>
              <button
                onClick={() => fetchIngredients(itemDetails.name)}
                className="bg-greenCustom text-white p-2 rounded-lg shadow-lg shadow-greenCustom hover:bg-light-green-700 hover:shadow-light-green-700 transition duration-500"
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
