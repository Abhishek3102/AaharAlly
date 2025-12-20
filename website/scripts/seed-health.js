require('dotenv').config({ path: '.env' });
const mongoose = require('mongoose');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// --- SCHEMAS (Inline to avoid import issues) ---
const FoodSchema = new mongoose.Schema({
    name: String,
    description: String,
    ingredients: String,
    image: String,
    category: String,
    meal_type: String,
});
// Handle potential model overwrite
const Food = mongoose.models.Food || mongoose.model("Food", FoodSchema);

const HealthCacheSchema = new mongoose.Schema({
    condition: { type: String, required: true, unique: true },
    safe_food_ids: [{ type: String, required: true }],
    last_updated: { type: Date, default: Date.now }
});
const HealthCache = mongoose.models.HealthCache || mongoose.model("HealthCache", HealthCacheSchema);

// --- CONFIG ---
const MONGO_URL = process.env.MONGO_URL;
const GENAI_API_KEY = process.env.NEXT_PUBLIC_GENAI;

if (!MONGO_URL) { console.error("Missing MONGO_URL"); process.exit(1); }
if (!GENAI_API_KEY) { console.error("Missing NEXT_PUBLIC_GENAI"); process.exit(1); }

const genAI = new GoogleGenerativeAI(GENAI_API_KEY);

const DISEASES = [
    "Diabetes", "Hypoglycemia", "Gastroparesis", "IBS", "Peptic Ulcer",
    "Hyperthyroidism", "Kidney Disease", "Cystic Fibrosis", "Addison's Disease"
];

// --- GEMINI LOGIC ---
async function filterFoodsByCondition(foods, condition) {
    if (!condition) return foods;
    const model = genAI.getGenerativeModel({ model: "gemini-flash-lite-latest" });

    const foodList = foods.map(f => ({
        id: f._id,
        name: f.name,
        description: f.description,
        ingredients: f.ingredients || "",
    }));

    const prompt = `
    You are a nutritionist assistant. I have a list of foods and a specific health condition: "${condition}".
    Task: Identify which of the following foods are GENERALLY considered safe or acceptable for someone with this condition.
    Strictly exclude foods that are known triggers or harmful for "${condition}".
    Input Foods: ${JSON.stringify(foodList)}
    Output Format: Return ONLY a raw JSON array of strings, where each string is the "id" of a safe food item.
    Do not include markdown. Example: ["id1", "id2"]
    `;

    try {
        const result = await model.generateContent(prompt);
        let text = result.response.text();
        text = text.replace(/```json/g, "").replace(/```/g, "").trim();
        const safeIds = JSON.parse(text);
        if (!Array.isArray(safeIds)) return [];
        
        return foods.filter(f => safeIds.includes(f._id.toString()));
    } catch (error) {
        console.error(`Gemini Error for ${condition}:`, error.message);
        return [];
    }
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// --- MAIN ---
async function main() {
    try {
        console.log("Connecting to MongoDB...");
        await mongoose.connect(MONGO_URL);
        console.log("Connected.");

        const allFoods = await Food.find({});
        console.log(`Found ${allFoods.length} foods.`);

        for (const disease of DISEASES) {
            console.log(`Processing: ${disease}...`);
            const safeFoods = await filterFoodsByCondition(allFoods, disease);
            const safeIds = safeFoods.map(f => f._id.toString());

            await HealthCache.findOneAndUpdate(
                { condition: disease },
                { safe_food_ids: safeIds, last_updated: new Date() },
                { upsert: true, new: true }
            );
            console.log(`  -> Saved ${safeIds.length} safe items.`);
            await delay(2000); // 2 second delay to avoid Rate Limits
        }

        console.log("All done!");
        process.exit(0);
    } catch (e) {
        console.error("Script Error:", e);
        process.exit(1);
    }
}

main();
