
const mongoose = require('mongoose');
const MONGO_URL = "mongodb+srv://worker48work_db_user:QuYZjwEN6OMWxHaC@aaharally-new.pgkhjhq.mongodb.net/?appName=aaharally-new";

async function run() {
    try {
        await mongoose.connect(MONGO_URL);
        console.log("Connected to DB:", mongoose.connection.name);
        
        console.log("Deleting old healthcaches...");
        const res = await mongoose.connection.collection('healthcaches').deleteMany({});
        console.log(`Deleted ${res.deletedCount} entries.`);
        
        console.log("Ready for fresh ML Trigger.");
    } catch (e) {
        console.error(e);
    } finally {
        await mongoose.disconnect();
    }
}
run();
