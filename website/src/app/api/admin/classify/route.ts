import { NextResponse } from "next/server";
import { mongoConnect } from "@/app/utils/feature";
import { food } from "@/app/models/Food";
import { HealthCache } from "@/app/models/HealthCache";
import { filterFoodsByCondition } from "@/app/utils/gemini";

// Hardcoded list of diseases matching the Frontend
const DISEASES = [
    "Diabetes",
    "Hypoglycemia",
    "Gastroparesis",
    "IBS",
    "Peptic Ulcer",
    "Hyperthyroidism",
    "Kidney Disease",
    "Cystic Fibrosis",
    "Addison's Disease"
];

export async function POST(req: Request) {
    try {
        await mongoConnect();

        // 1. Fetch ALL foods (lean for performance)
        const allFoods = await food.find({}).lean();
        console.log(`Classifying ${allFoods.length} foods...`);

        const results = [];

        // 2. Process each disease
        for (const disease of DISEASES) {
            console.log(`Processing: ${disease}`);

            // Ask Gemini which foods are safe
            // This returns the full food objects, we just need IDs
            const safeFoods = await filterFoodsByCondition(allFoods, disease);
            const safeIds = safeFoods.map((f: any) => f._id.toString());

            // 3. Update Cache
            const cacheEntry = await HealthCache.findOneAndUpdate(
                { condition: disease },
                {
                    safe_food_ids: safeIds,
                    last_updated: new Date()
                },
                { upsert: true, new: true }
            );
            results.push({ disease, count: safeIds.length });
        }

        return NextResponse.json({ success: true, results }, { status: 200 });

    } catch (error: any) {
        console.error("Classification Error:", error);
        return NextResponse.json({ success: false, error: error.message }, { status: 500 });
    }
}
