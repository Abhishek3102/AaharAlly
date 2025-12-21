import { userData } from "@/app/models/UserData";
import { food } from "../../models/Food";
import FoodPreference from "../../models/FoodPreferenceSchema";
import { mongoConnect } from "../../utils/feature";
import { NextResponse } from "next/server";
import { filterFoodsByCondition } from "../../utils/gemini";
import { HealthCache } from "../../models/HealthCache";
import { qdrant, FOOD_COLLECTION } from "@/lib/qdrant";
import { generateEmbedding } from "../../utils/gemini";


export async function GET(req: Request) {
    console.log("!!! API HIT - Users/route.ts !!!"); // TRACER LOG
    const url = new URL(req.url);
    let semanticScoreMap: Map<string, number> | null = null;
    const ageParam = url.searchParams.get('age');
    const id = url.searchParams.get('id');
    const regions = url.searchParams.get('regions'); // Allow multiple regions
    const categoryParam = url.searchParams.get('categories'); // Allow multiple categories
    const meal_type = url.searchParams.get('meal_type');
    const search = url.searchParams.get('search');
    const health_condition = url.searchParams.get('health_condition');

    try {
        await mongoConnect();

        // Find specific item by ID
        if (id) {
            const data = await food.findById({ _id: id });
            return NextResponse.json({ data, success: true }, { status: 200 });
        }

        const conditions: any = {};
        let categoriesArray: any[] = [];
        let regionsArray: string[] = [];

        // If age is provided, find the category that matches the age range
        if (ageParam) {
            const age = parseInt(ageParam, 10);
            const ageBasedCategory = await FoodPreference.findOne({
                minAge: { $lte: age },
                maxAge: { $gte: age }
            }).select("food_category");

            if (ageBasedCategory) {
                categoriesArray.push(ageBasedCategory.food_category);
            }
        }

        // Parse regions and set to an array for aggregation
        if (regions) {
            regionsArray = regions.split(',').map(item => item.trim());
        }

        // If regions are provided, calculate most consumed categories in the regions
        if (regionsArray.length > 0) {
            const consumptionData = await userData.aggregate([
                { $match: { region: { $in: regionsArray } } }, // Match any of the provided regions
                {
                    $group: {
                        _id: { region: '$region', meal_category: '$meal_category' },
                        categoryCount: { $sum: 1 }
                    }
                },
                {
                    $group: {
                        _id: '$_id.region',
                        totalOrdersInRegion: { $sum: '$categoryCount' },
                        categories: {
                            $push: {
                                meal_category: '$_id.meal_category',
                                categoryCount: '$categoryCount'
                            }
                        }
                    }
                },
                {
                    $project: {
                        _id: 0,
                        region: '$_id',
                        totalOrdersInRegion: 1,
                        categories: {
                            $map: {
                                input: '$categories',
                                as: 'category',
                                in: {
                                    meal_category: '$$category.meal_category',
                                    categoryCount: '$$category.categoryCount',
                                    percentage: {
                                        $multiply: [
                                            { $divide: ['$$category.categoryCount', '$totalOrdersInRegion'] },
                                            100
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            ]);

            // Collect the most consumed categories from the aggregation
            if (consumptionData.length > 0) {
                consumptionData.forEach(data => {
                    const mostConsumedCategory = data.categories.reduce((prev: any, current: any) => {
                        return prev.categoryCount > current.categoryCount ? prev : current;
                    });
                    categoriesArray.push(mostConsumedCategory.meal_category);
                });
            }
        }

        // If specific categories are provided in query, add to the categoriesArray
        if (categoryParam) {
            const incomingCategoriesArray = categoryParam.split(',').map(item => item.trim());
            categoriesArray = [...new Set([...categoriesArray, ...incomingCategoriesArray])]; // Combine and deduplicate
        }

        // Filter for category if we have any from the previous checks
        if (categoriesArray.length > 0) {
            conditions.category = { $in: categoriesArray };
        }

        if (meal_type) {
            conditions.meal_type = meal_type;
        }

        // Search functionality
        // Search functionality
        if (search) {
            let usedSemanticSearch = false;
            // semanticScoreMap is now defined at top level

            // 1. Try Semantic Search
            // REMOVED TRY/CATCH for Debugging
            // Heuristic: If search has spaces or > 3 chars, try semantic
            if (search.length > 2) {
                console.log(`Checking embedding for: '${search}'`);
                const embedding = await generateEmbedding(search);
                if (embedding) {
                    console.log("generated embedding, searching qdrant...");
                    const searchResults = await qdrant.search(FOOD_COLLECTION, {
                        vector: embedding,
                        limit: 10,
                        with_payload: true,
                        score_threshold: 0.60
                    });

                    console.log("--- Semantic Search Results ---");
                    searchResults.forEach((res: any) => {
                        console.log(`Item: ${res.payload.name} | Score: ${res.score}`);
                    });
                    console.log("-------------------------------");

                    const scoreMap: Record<string, number> = {};
                    searchResults.forEach((res: any) => {
                        if (res.payload && res.payload.mongo_id) {
                            scoreMap[res.payload.mongo_id] = res.score;
                        }
                    });

                    const ids = searchResults
                        .filter((res: any) => res.payload && res.payload.mongo_id)
                        .map((res: any) => res.payload.mongo_id);

                    // If we found semantic matches, restrict DB query to these IDs
                    if (ids.length > 0) {
                        conditions._id = { $in: ids };
                        usedSemanticSearch = true;
                        semanticScoreMap = new Map(Object.entries(scoreMap)); // Initialize the map here
                    } else {
                        console.log("Semantic search returned 0 results.");
                    }
                } else {
                    console.error("Failed to generate embedding for query.");
                }
            }

            // 2. Fallback to Regex if Semantic Search didn't run or found nothing
            // (Or maybe we want to OR them? For now, strict Semantic if succeeds is better for "smart" feel)
            if (!usedSemanticSearch) {
                const searchRegex = new RegExp(search, 'i');
                conditions.$or = [
                    { name: searchRegex },
                    { description: searchRegex },
                    { category: searchRegex }
                ];
            }
        }



        // Find and return data based on final conditions
        let data = await food.find(conditions).lean(); // Use lean() to return headers

        // Sort and Attach Scores
        if (search && conditions._id && conditions._id.$in && semanticScoreMap) {
            // 1. Sort
            const idList = conditions._id.$in;
            data.sort((a: any, b: any) => {
                return idList.indexOf(a._id.toString()) - idList.indexOf(b._id.toString());
            });

            // 2. Attach Scores (We need to access scoreMap from the upper scope)
            // To do this cleanly without a massive refactor, we can re-map here if we had the map.
            // But 'scoreMap' is inside the try block.
            // Let's rely on the fact that if we sorted by ID list, we can infer relevance, but User wants NUMBERS.
            // We must move scoreMap to outer scope.
            data = data.map((item: any) => {
                const score = semanticScoreMap?.get(item._id.toString());
                return score !== undefined ? { ...item, score } : item;
            });
        }

        // --- GEMINI FILTERING (CACHED) ---
        if (health_condition) {
            // 1. Fetch pre-calculated safe IDs
            const cache = await HealthCache.findOne({ condition: health_condition });

            if (cache && cache.safe_food_ids) {
                // 2. Filter the current result set
                // Convert IDs to string for comparison or use includes
                const safeSet = new Set(cache.safe_food_ids.map((id: any) => id.toString()));
                data = data.filter((item: any) => safeSet.has(item._id.toString()));
            } else {
                // Feature fallback: If cache empty, maybe return nothing or all?
                // User expects valid filtering. If we haven't run the classifier, returning all is dangerous for health.
                // We return empty array with a log.
                console.warn(`No health cache found for: ${health_condition}`);
                data = [];
            }
        }

        return NextResponse.json({ data, success: true }, { status: 200 });
    } catch (err: any) {
        console.error("API Error:", err);
        return NextResponse.json({ message: `Error processing request: ${err.message}`, success: false }, { status: 500 });
    }
}

