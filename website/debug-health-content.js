
const mongoose = require('mongoose');

// Vercel DB URL (worker48)
const MONGO_URL = "mongodb+srv://worker48work_db_user:QuYZjwEN6OMWxHaC@aaharally-new.pgkhjhq.mongodb.net/?appName=aaharally-new";

async function run() {
    try {
        await mongoose.connect(MONGO_URL);
        console.log("Connected to DB:", mongoose.connection.name);

        const HealthCache = mongoose.connection.collection('healthcaches');
        const count = await HealthCache.countDocuments();
        console.log(`Total Health Caches: ${count}`);

        const all = await HealthCache.find({}).toArray();
        all.forEach(doc => {
            console.log(`\nCondition: "${doc.condition}"`);
            console.log(`Updated: ${doc.last_updated}`);
            console.log(`Safe IDs Count: ${doc.safe_food_ids ? doc.safe_food_ids.length : 0}`);
            if (doc.safe_food_ids && doc.safe_food_ids.length > 0) {
                console.log(`Sample ID: ${doc.safe_food_ids[0]}`);
            }
        });

        // Also check if these IDs exist in Food
        if (all.length > 0 && all[0].safe_food_ids.length > 0) {
            const sampleId = all[0].safe_food_ids[0];
            const Food = mongoose.connection.collection('foods');
            // Try string and ObjectId
            let found = await Food.findOne({ _id: sampleId });
             if (!found) {
                 try {
                    found = await Food.findOne({ _id: new mongoose.Types.ObjectId(sampleId) });
                 } catch(e) {}
             }
            
            console.log(`\nIntegrity Check for ID ${sampleId}: ${found ? "FOUND in Food" : "NOT FOUND in Food"}`);
        }

    } catch (e) {
        console.error(e);
    } finally {
        await mongoose.disconnect();
    }
}

run();
