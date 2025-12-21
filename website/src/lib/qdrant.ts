import { QdrantClient } from "@qdrant/js-client-rest";

if (!process.env.QDRANT_URL || !process.env.QDRANT_API_KEY) {
    throw new Error("QDRANT_URL and QDRANT_API_KEY must be defined");
}

export const qdrant = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});

export const FOOD_COLLECTION = "food_items";
