
require('dotenv').config({ path: '.env' });
const mongoose = require('mongoose');

// Define inline schemas to avoid import issues
const FoodSchema = new mongoose.Schema({ name: String, menu: [{ type: mongoose.Schema.Types.ObjectId, ref: 'food' }] });
// Note: Restaurant schema definition here is simplified for query
const RestaurantSchema = new mongoose.Schema({ name: String, menu: [{ type: mongoose.Schema.Types.ObjectId, ref: 'food' }] });

const Food = mongoose.models.food || mongoose.model('food', FoodSchema);
const Restaurant = mongoose.models.Restaurant || mongoose.model('Restaurant', RestaurantSchema);

async function run() {
    try {
        await mongoose.connect(process.env.MONGO_URL);
        console.log("Connected to MongoDB");

        const items = await Food.find({ name: /pancake/i });
        console.log(`Found ${items.length} items matching 'pancake':`);
        items.forEach(i => console.log(`- ${i.name} (${i._id})`));

        if (items.length > 0) {
            const ids = items.map(i => i._id);
            const hotels = await Restaurant.find({ menu: { $in: ids } });
            console.log(`\nFound in ${hotels.length} hotels:`);
            hotels.forEach(h => console.log(`- ${h.name}`));
        } else {
            console.log("No pancakes found in Food collection.");
        }

    } catch (e) {
        console.error(e);
    } finally {
        await mongoose.disconnect();
    }
}

run();
