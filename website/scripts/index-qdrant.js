const mongoose = require('mongoose');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { QdrantClient } = require('@qdrant/js-client-rest');
const dotenv = require('dotenv');
const { join } = require('path');

// Load env parameters
dotenv.config({ path: join(__dirname, '../.env') });

const MONGO_URL = process.env.MONGO_URL;
const NEXT_PUBLIC_GENAI = process.env.NEXT_PUBLIC_GENAI;
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const FOOD_COLLECTION = "food_items";

if (!MONGO_URL || !NEXT_PUBLIC_GENAI || !QDRANT_URL || !QDRANT_API_KEY) {
    console.error("Missing Environment Variables");
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

const genAI = new GoogleGenerativeAI(NEXT_PUBLIC_GENAI);
const model = genAI.getGenerativeModel({ model: "gemini-embedding-001" });
const qdrant = new QdrantClient({ url: QDRANT_URL, apiKey: QDRANT_API_KEY });

// Delay helper to avoid rate limits
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function generateEmbedding(text) {
    try {
        const result = await model.embedContent(text);
        return result.embedding.values;
    } catch (e) {
        console.error("Embedding Error:", e.message);
        return null;
    }
}

async function seed() {
    try {
        console.log("Connecting to MongoDB...");
        await mongoose.connect(MONGO_URL);
        console.log("Connected.");

        // 1. Initial Qdrant Setup
        const collections = await qdrant.getCollections();
        const exists = collections.collections.some(c => c.name === FOOD_COLLECTION);
        
        // 200 items logic
        console.log("Fetching foods...");
        const foods = await Food.find({});
        console.log(`Found ${foods.length} foods to index.`);

        if (foods.length === 0) {
            console.log("No foods found.");
            process.exit(0);
        }

        // Generate one embedding to detect size
        console.log("Detecting vector dimension...");
        const firstFood = foods[0];
        const sampleText = `${firstFood.name}. ${firstFood.description}`;
        const sampleVector = await generateEmbedding(sampleText);
        
        if (!sampleVector) {
             console.error("Could not generate sample embedding. Aborting.");
             process.exit(1);
        }
        
        const detectedSize = sampleVector.length;
        console.log(`Detected Vector Size: ${detectedSize}`);

        if (exists) {
            console.log(`Deleting existing collection ${FOOD_COLLECTION} to ensure schema match...`);
            await qdrant.deleteCollection(FOOD_COLLECTION);
        }

        console.log(`Creating Qdrant collection: ${FOOD_COLLECTION} with size ${detectedSize}...`);
        await qdrant.createCollection(FOOD_COLLECTION, {
            vectors: { size: detectedSize, distance: 'Cosine' },
        });

        let successCount = 0;

        for (let i = 0; i < foods.length; i++) {
            const food = foods[i];

            
            // Text to embed: Rich context
            const textToEmbed = `${food.name}. ${food.description}. Category: ${food.category}. Meal: ${food.meal_type}.`;
            
            // Retry logic for embedding
            let vector = null;
            let retries = 3;
            while(retries > 0 && !vector) {
                vector = await generateEmbedding(textToEmbed);
                if(!vector) {
                    await delay(2000); // Wait on error
                    retries--;
                }
            }

            if (!vector) {
                console.error(`Skipping ${food.name} (Embedding Failed)`);
                continue;
            }

            // Debug first vector
            if (i === 0) {
                console.log(`Vector Size Detected: ${vector.length}`);
            }

            // Convert MongoID (24 hex) to Fake UUID (32 hex)
            // Valid UUID is 32 hex chars (with dashes usually, but Qdrant accepts clean hex)
            // Padding with 8 zeros to make length 32
            const qdrantId = '00000000' + food._id.toString(); 
            // Format: 00000000-xxxx-xxxx-xxxx-xxxxxxxxxxxx (Strictly Qdrant might need dashes for string UUID validation)
            // Let's format it properly as UUID: 8-4-4-4-12
            // 00000000-aaaa-bbbb-cccc-dddddddddddd
            // MongolID: aaaa bbbb cccc dddddddddddd (24 chars)
            // So: 00000000 - first 4 of mongo - next 4 of mongo - next 4 of mongo - last 12 of mongo
            const mId = food._id.toString();
            const formattedId = `00000000-${mId.substring(0,4)}-${mId.substring(4,8)}-${mId.substring(8,12)}-${mId.substring(12,24)}`;

            try {
                await qdrant.upsert(FOOD_COLLECTION, {
                    wait: true,
                    points: [{
                        id: formattedId,
                        vector: vector,
                        payload: {
                            mongo_id: food._id.toString(),
                            name: food.name,
                            category: food.category
                        }
                    }]
                });
                successCount++;
                process.stdout.write(`\rIndexed ${successCount}/${foods.length}: ${food.name.substring(0,20)}...`);
            } catch (err) {
                console.error(`\nFailed to upsert ${food.name}:`, err.message);
            }

            // Rate Limit Protection (free tier)
            await delay(500); 
        }

        console.log("\n\nIndexing Complete!");
        process.exit(0);

    } catch (e) {
        console.error("Script Error:", e);
        process.exit(1);
    }
}

seed();
