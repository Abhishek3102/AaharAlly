const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

// Load env parameters
dotenv.config({ path: path.join(__dirname, '../.env') });

const MONGO_URL = process.env.MONGO_URL;

if (!MONGO_URL) {
    console.error("Missing MONGO_URL");
    process.exit(1);
}

const foodSchema = new mongoose.Schema({
    name: String,
    description: String,
    category: String,
    meal_type: String,
    price: String,
    image: String,
    rating: String
}, { strict: false });

const Food = mongoose.models.food || mongoose.model('food', foodSchema);

async function seed() {
    try {
        console.log("Connecting to MongoDB...");
        await mongoose.connect(MONGO_URL);
        console.log("Connected.");
        
        const jsonPath = path.join(__dirname, '../src/app/hotelUser/components/food_items.json');
        console.log(`Reading data from ${jsonPath}...`);
        
        const rawData = fs.readFileSync(jsonPath);
        const foods = JSON.parse(rawData);
        
        console.log(`Found ${foods.length} items in JSON file.`);
        
        console.log("Deleting existing food items in MongoDB...");
        await Food.deleteMany({});
        console.log("Cleared old data.");
        
        console.log("Inserting new items...");
        // Batch insert for efficiency
        await Food.insertMany(foods);
        
        console.log(`Successfully inserted ${foods.length} items!`);
        process.exit(0);

    } catch (e) {
        console.error("Seeding Error:", e);
        process.exit(1);
    }
}

seed();
