
require('dotenv').config({ path: '.env' });
const mongoose = require('mongoose');
const { GoogleGenerativeAI } = require("@google/generative-ai");

// Mock Qdrant for this test (or use if possible, but let's test logic flow first)
// Actually, let's test the ROUTE logic logic if possible.
// But I can't import the route easily.
// I will simulate the logic in this script.

async function run() {
    console.log("Simulating Logic...");
    const search = "pancakes";
    
    // Logic from route.ts:
    // 1. Regex
    const searchRegex = new RegExp(search, 'i');
    console.log(`Regex: ${searchRegex}`);
    
    await mongoose.connect(process.env.MONGO_URL);
    const FoodSchema = new mongoose.Schema({ name: String, description: String, category: String }, { strict: false });
    const Food = mongoose.models.food || mongoose.model('food', FoodSchema);

    const results = await Food.find({
        $or: [
            { name: searchRegex },
            { description: searchRegex },
            { category: searchRegex }
        ]
    }).lean();

    console.log(`Regex Matches found: ${results.length}`);
    results.forEach(r => console.log(`- ${r.name}`));

    if (results.length === 0) {
        console.log("Regex found NOTHING. If Vector Search also fails, result is 0.");
    }

    // Checking if Vector Search (mocked) would be used.
    if (search.length > 2) {
        console.log("Vector Search would be triggered.");
    }
    
    await mongoose.disconnect();
}

run();
