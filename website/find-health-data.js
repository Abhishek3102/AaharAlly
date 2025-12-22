
const mongoose = require('mongoose');

// Vercel DB URL (worker48)
const MONGO_URL = "mongodb+srv://worker48work_db_user:QuYZjwEN6OMWxHaC@aaharally-new.pgkhjhq.mongodb.net/?appName=aaharally-new";

async function run() {
    try {
        console.log("Connecting...");
        await mongoose.connect(MONGO_URL);
        
        const admin = new mongoose.mongo.Admin(mongoose.connection.db);
        const dbs = await admin.listDatabases();
        
        console.log("Databases found:", dbs.databases.map(d => d.name));

        const targetInfo = [];

        for (const dbInfo of dbs.databases) {
            const dbName = dbInfo.name;
            if (['local', 'config', 'admin'].includes(dbName)) continue;

            console.log(`Checking DB: ${dbName}`);
            const db = mongoose.connection.useDb(dbName);
            const collections = await db.db.listCollections().toArray();
            
            const hasHealth = collections.find(c => c.name.toLowerCase().includes('health'));
            if (hasHealth) {
                const count = await db.collection(hasHealth.name).countDocuments();
                console.log(`  FOUND COLLECTION: ${hasHealth.name} (Count: ${count})`);
                targetInfo.push({ db: dbName, col: hasHealth.name, count });
            }
        }

        console.log("\n--- SUMMARY ---");
        if (targetInfo.length > 0) {
            console.log("HealthCache locations:", targetInfo);
        } else {
            console.log("CRITICAL: 'healthcaches' collection NOT FOUND in ANY database.");
        }

    } catch (e) {
        console.error(e);
    } finally {
        await mongoose.disconnect();
    }
}

run();
