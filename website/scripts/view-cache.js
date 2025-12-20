require('dotenv').config({ path: '.env' });
const mongoose = require('mongoose');

const HealthCacheSchema = new mongoose.Schema({
    condition: { type: String, required: true, unique: true },
    safe_food_ids: [{ type: String, required: true }],
    last_updated: { type: Date, default: Date.now }
});
const HealthCache = mongoose.models.HealthCache || mongoose.model("HealthCache", HealthCacheSchema);

const MONGO_URL = process.env.MONGO_URL;

if (!MONGO_URL) { console.error("Missing MONGO_URL"); process.exit(1); }

const fs = require('fs');

async function main() {
    try {
        await mongoose.connect(MONGO_URL);
        const count = await HealthCache.countDocuments({});
        const caches = await HealthCache.find({});
        
        let output = `Connected to DB: ${mongoose.connection.name}\n`;
        output += `Total Documents: ${count}\n\n`;
        
        caches.forEach(c => {
            output += `Condition: ${c.condition}\n`;
            output += `Safe Foods: ${c.safe_food_ids ? c.safe_food_ids.length : 0}\n`;
            output += `Sample IDs: ${c.safe_food_ids ? c.safe_food_ids.slice(0, 5).join(', ') : 'None'}\n`;
            output += "----------------\n";
        });

        fs.writeFileSync('scripts/output.txt', output);
        console.log("Output written to scripts/output.txt");
        process.exit(0);
    } catch (e) {
        console.error(e);
        process.exit(1);
    }
}

main();
