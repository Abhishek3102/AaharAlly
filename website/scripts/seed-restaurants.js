
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from the parent directory (website/.env)
dotenv.config({ path: path.resolve(__dirname, '../.env') });

import mongoose from 'mongoose';

const MONGO_URL = process.env.MONGO_URL;

if (!MONGO_URL) {
    console.error("Missing MONGO_URL");
    process.exit(1);
}

// Model Definitions (Inline to avoid TS import issues in script)
const FoodSchema = new mongoose.Schema({
    name: String, image: String, rating: String, price: String, category: String, description: String, meal_type: String
});
const Food = mongoose.models.food || mongoose.model('food', FoodSchema);

const RestaurantSchema = new mongoose.Schema({
    name: String, description: String, address: String,
    location: { lat: Number, lng: Number },
    image: String,
    owner: { name: String, image: String },
    rating: Number,
    menu: [{ type: mongoose.Schema.Types.ObjectId, ref: 'food' }]
}, { timestamps: true });
const Restaurant = mongoose.models.Restaurant || mongoose.model('Restaurant', RestaurantSchema);

// Mumbai Data
const HOTELS = [
    {
        name: "Taj Lands End",
        description: "Luxury hotel overlooking the Arabian Sea, known for exquisite dining.",
        address: "Bandstand, BJ Road, Mount Mary, Bandra West, Mumbai, Maharashtra 400050",
        location: { lat: 19.0433, lng: 72.8197 }, // Bandra
        image: "https://images.unsplash.com/photo-1566073771259-6a8506099945?q=80&w=2070&auto=format&fit=crop",
        owner: { name: "Dr. Vivek DeoDeshmukh", image: "/owners/deo.jpg" },
        rating: 4.8
    },
    {
        name: "The Oberoi",
        description: "Elegant hotel with panoramic ocean views and world-class cuisine.",
        address: "Nariman Point, Mumbai, Maharashtra 400021",
        location: { lat: 18.9272, lng: 72.8207 }, // Nariman Point
        image: "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?q=80&w=2070&auto=format&fit=crop",
        owner: { name: "Dr. Dipali Bhole", image: "/owners/dips.jpg" },
        rating: 4.9
    },
    {
        name: "JW Marriott Juhu",
        description: "Beachfront 5-star hotel famous for its celebrity sightings and Sunday brunch.",
        address: "Juhu Tara Road, Mumbai, Maharashtra 400049",
        location: { lat: 19.1026, lng: 72.8258 }, // Juhu
        image: "https://images.unsplash.com/photo-1571896349842-68c894913dbb?q=80&w=2070&auto=format&fit=crop",
        owner: { name: "Dr. Hezal Lopes", image: "/owners/hezu.jpg" },
        rating: 4.7
    },
    {
        name: "The Westin Mumbai Garden City",
        description: "Modern high-rise hotel offering a tranquil retreat in the bustle of Goregaon.",
        address: "Goregaon East, International Business Park, Mumbai, Maharashtra 400063",
        location: { lat: 19.1727, lng: 72.8580 }, // Goregaon
        image: "https://images.unsplash.com/photo-1455587734955-081b22074882?q=80&w=1920&auto=format&fit=crop",
        owner: { name: "Dr. Shilpa Satre", image: "/owners/shilpa.jpg" },
        rating: 4.6
    },
    {
        name: "St. Regis Mumbai",
        description: "The tallest hotel tower in the city, offering supreme luxury and shopping access.",
        address: "462, Senapati Bapat Marg, Lower Parel, Mumbai, Maharashtra 400013",
        location: { lat: 18.9937, lng: 72.8249 }, // Lower Parel
        image: "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?q=80&w=2070&auto=format&fit=crop",
        owner: { name: "Dr. Swapnil Gharat", image: "/owners/swapnil.jpg" },
        rating: 4.9
    }
];

async function seed() {
    try {
        console.log("Connecting to MongoDB...");
        await mongoose.connect(MONGO_URL);
        console.log("Connected.");

        console.log("Fetching Foods...");
        const allFoods = await Food.find({});
        if (allFoods.length === 0) {
            console.error("No foods found! Run migration first.");
            process.exit(1);
        }
        console.log(`Found ${allFoods.length} foods from database.`);

        console.log("Clearing existing restaurants...");
        await Restaurant.deleteMany({});

        console.log("Seeding Restaurants...");
        for (const hotel of HOTELS) {
            // Assign 15 random foods to this hotel
            const shuffled = allFoods.sort(() => 0.5 - Math.random());
            const menu = shuffled.slice(0, 15).map(f => f._id);

            await Restaurant.create({
                ...hotel,
                menu: menu
            });
            console.log(`Created: ${hotel.name} with ${menu.length} items.`);
        }

        console.log("\nSeeding Complete! 🏨");
        process.exit(0);

    } catch (e) {
        console.error(e);
        process.exit(1);
    }
}

seed();
