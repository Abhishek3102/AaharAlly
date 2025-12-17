"use client";
import { useUser } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";
interface AgeModalProps {
  isModalOpen: boolean;
  setIsModalOpen: (open: boolean) => void;
}

const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GENAI!);
const AgeModal: React.FC<AgeModalProps> = ({ isModalOpen, setIsModalOpen }) => {
  const [age, setAge] = useState("");
  const { user } = useUser();
  const [selectedDisease, setSelectedDisease] = useState("");
  const [loading, setLoading] = useState(false);

  const diseases = [
    { label: "Diabetes", value: "Diabetes" },
    { label: "Hypoglycemia", value: "Hypoglycemia" },
    { label: "Gastroparesis", value: "Gastroparesis" },
    { label: "IBS", value: "IBS" },
    { label: "Peptic Ulcer", value: "Peptic Ulcer" },
    { label: "Hyperthyroidism", value: "Hyperthyroidism" },
    { label: "Kidney Disease", value: "Kidney Disease" },
    { label: "Cystic Fibrosis", value: "Cystic Fibrosis" },
    { label: "Addison's Disease", value: "Addison's Disease" },
  ];

  useEffect(() => {
    // const age = user?.unsafeMetadata.age;
  }, [user]);

  async function call(disease: string) {
    setLoading(true);
    const model = genAI.getGenerativeModel({
      model: "gemini-flash-lite-latest",
    });
    const prompt = `I have ${disease}. Suggest me only 20 best food items from this food item array: [\n  "Tofu Scramble",\n  "Vegan Burger",\n  "Falafel",\n  "Vegan Sushi",\n  "Vegan Pizza",\n  "Vegan Chili",\n  "Vegan Pasta",\n  "Vegan Salad",\n  "Vegan Smoothie",\n  "Vegan Ice Cream",\n  "Jackfruit Tacos",\n  "Vegan Curry",\n  "Vegan Ramen",\n  "Vegan Stir-Fry",\n  "Vegan Quesadillas",\n  "Vegan Nachos",\n  "Vegan Pancakes",\n  "Vegan Waffles",\n  "Vegan Shepherd's Pie",\n  "Vegan Lasagna",\n  "Chocolate Chip Cookies",\n  "Cheesecake",\n  "Brownie",\n  "Ice Cream Sundae",\n  "Apple Pie",\n  "Tiramisu",\n  "Fruit Tart",\n  "Crème brûlée",\n  "Pancakes",\n  "Waffles",\n  "Donut",\n  "Cupcake",\n  "Mousse",\n  "Flan",\n  "Pie",\n  "Pudding",\n  "Ice Cream",\n  "Sorbet",\n  "Cake",\n  "Macarons",\n  "Tacos",\n  "Hot Dogs",\n  "Pizza by the Slice",\n  "Falafel",\n  "Empanadas",\n  "Chicken and Waffles",\n  "Ice Cream",\n  "Fried Chicken",\n  "Sushi",\n  "Popcorn",\n  "Pretzels",\n  "Waffles",\n  "Grilled Cheese",\n  "Poutine",\n  "Spring Rolls",\n  "Arepas",\n  "Crepes",\n  "Shawarma",\n  "Chicken Skewers",\n  "Fruit Smoothies",\n  "Spicy Kimchi Fried Rice",\n  "Buffalo Wings",\n  "Spicy Sichuan Noodles",\n  "Jalapeño Popper Pizza",\n  "Spicy Tuna Roll",\n  "Jerk Chicken",\n  "Spicy Thai Curry",\n  "Nashville Hot Chicken Sandwich",\n  "Spicy Shrimp Tacos",\n  "Sriracha Mayo Fries",\n  "Sambal Fried Rice",\n  "Ghost Pepper Wings",\n  "Spicy Korean BBQ",\n  "Spicy Arrabiata Pasta",\n  "Spicy Beef & Broccoli",\n  "Chilli Con Carne",\n  "Spicy Tuna Poke Bowl",\n  "Spicy Chorizo Quesadilla",\n  "Spicy Shrimp Pad Thai",\n  "Spicy Black Bean Burgers",\n  "Mac and Cheese",\n  "Cheese Pizza",\n  "Grilled Cheese Sandwich",\n  "Cheese Quesadilla",\n  "Cheesy Fries",\n  "Cheese Fondue",\n  "Cheese Curds",\n  "Cheesy Bread",\n  "Cheese and Broccoli Soup",\n  "Cheesecake",\n  "Mozzarella Sticks",\n  "Cheesy Potatoes",\n  "Macaroni and Cheese Bites",\n  "Cheese Ravioli",\n  "Cheese Tortellini",\n  "Cheese Stuffed Peppers",\n  "Cheese Ball",\n  "Cheese and Sausage Pizza",\n  "Cheesy Chicken Alfredo",\n  "Nachos",\n  "Kung Pao Chicken",\n  "Mapo Tofu",\n  "Chow Mein",\n  "Egg Rolls",\n  "Sweet and Sour Pork",\n  "General Tso's Chicken",\n  "Fried Rice",\n  "Wonton Soup",\n  "Peking Duck",\n  "Dim Sum",\n  "Spring Rolls",\n  "Moo Shu Pork",\n  "Sesame Chicken",\n  "Beef and Broccoli",\n  "Dan Dan Noodles",\n  "Hot and Sour Soup",\n  "Szechuan Green Beans",\n  "Steamed Fish",\n  "Lo Mein",\n  "Pot Stickers",\n  "Butter Chicken",\n  "Chicken Tikka Masala",\n  "Saag Paneer",\n  "Aloo Gobi",\n  "Chana Masala",\n  "Vindaloo",\n  "Malai Kofta",\n  " Rogan Josh",\n  "Biryani",\n  "Palak Paneer",\n  "Tandoori Chicken",\n  "Daal Makhani",\n  "Kadai Paneer",\n  "Chicken Curry",\n  "Fish Curry",\n  "Lamb Curry",\n  "Matar Paneer",\n  "Baingan Bharta",\n  "Chicken Tikka",\n  "Vegetable Curry",\n  "Potato Chips",\n  "Popcorn",\n  "Trail Mix",\n  "Fruit Snacks",\n  "Pretzels",\n  "Chocolate Bar",\n  "Yogurt",\n  "Crackers",\n  "Fruit",\n  "Energy Bars",\n  "Rice Cakes",\n  "Beef Jerky",\n  "Cheese",\n  "Nuts",\n  "Hummus",\n  "Hard-Boiled Eggs",\n  "Frozen Yogurt",\n  "Smoothie",\n  "Granola Bars",\n  "Candy",\n  "Salmon",\n  "Tuna",\n  "Shrimp",\n  "Crab",\n  "Lobster",\n  "Scallops",\n  "Oysters",\n  "Mussels",\n  "Clams",\n  "Cod",\n  "Haddock",\n  "Halibut",\n  "Swordfish",\n  "Mahi-Mahi",\n  "Snapper",\n  "Grouper",\n  "Octopus",\n  "Squid",\n  "Clam Chowder",\n  "Sushi",\n  "Avocado",\n  "Salmon",\n  "Broccoli",\n  "Quinoa",\n  "Blueberries",\n  "Sweet Potatoes",\n  "Spinach",\n  "Chickpeas",\n  "Almonds",\n  "Brown Rice",\n  "Lentils",\n  "Edamame",\n  "Greek Yogurt",\n  "Apples",\n  "Eggs",\n  "Oats",\n  "Tomatoes",\n  "Kale",\n  "Bananas",\n  "Tofu\"\n]   which will be good for me also make sure to give me proper json containing list of items from this and not other things in output\n`;
    try {
      const result = await model.generateContent(prompt);
      const responseText = result.response.text();
      const jsonMatch = responseText.match(/(\[[\s\S]*?\])/);

      if (jsonMatch && jsonMatch[1]) {
        const jsonResponse = JSON.parse(jsonMatch[1]);
        console.log(JSON.stringify(jsonResponse, null, 2));
      } else {
        console.error("No valid JSON found in the response.");
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  }

  const handleDiseaseSubmit = async () => {
    if (selectedDisease) {
      await call(selectedDisease);
      setIsModalOpen(false);
    }
    if (age) {
      user?.update({
        unsafeMetadata: {
          age,
        },
      });
      setIsModalOpen(false);
    }
  };

  return (
    <>
      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-5 rounded-2xl shadow-lg w-full max-w-md">
            <h2 className="text-lg font-semibold mb-4">
              Enter Your Details
              <button
                onClick={() => setIsModalOpen(false)}
                className="float-right"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </h2>
            {!user?.unsafeMetadata?.age && (
              <>
                <input
                  type="number"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  className="border rounded-xl p-2 w-full mb-4"
                  placeholder="Your age"
                />
              </>
            )}
            <select
              value={selectedDisease}
              onChange={(e) => setSelectedDisease(e.target.value)}
              className="border rounded-xl p-2 w-full mb-4"
            >
              <option value="">Select a condition (if applicable)</option>
              {diseases.map((disease) => (
                <option key={disease.value} value={disease.value}>
                  {disease.label}
                </option>
              ))}
            </select>
            <button
              disabled={loading}
              onClick={handleDiseaseSubmit}
              className="bg-red-500 text-white p-2 rounded-xl hover:bg-red-600 transition duration-300 w-full"
            >
              {loading ? "Processing..." : "Get Food"}
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default AgeModal;
