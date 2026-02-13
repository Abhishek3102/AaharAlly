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
            let vectorIds: string[] = [];
            let scoreMap: Record<string, number> = {};

            // 1. Try Semantic Search
            if (search.length > 2) {
                try {
                    const embedding = await generateEmbedding(search);
                    if (embedding) {
                        const searchResults = await qdrant.search(FOOD_COLLECTION, {
                            vector: embedding,
                            limit: 15, // Increased limit
                            with_payload: true,
                            score_threshold: 0.60
                        });

                        searchResults.forEach((res: any) => {
                            if (res.payload && res.payload.mongo_id) {
                                vectorIds.push(res.payload.mongo_id);
                                scoreMap[res.payload.mongo_id] = res.score;
                            }
                        });

                        if (vectorIds.length > 0) {
                            semanticScoreMap = new Map(Object.entries(scoreMap));
                        }
                    }
                } catch (e) {
                    console.error("Vector search error", e);
                }
            }

            // 2. Hybrid Query Construction
            const searchRegex = new RegExp(search, 'i');
            const regexConditions = [
                { name: searchRegex },
                { description: searchRegex },
                { category: searchRegex }
            ];

            if (vectorIds.length > 0) {
                // If we have vector matches, we want (Vector IDs) OR (Regex Matches)
                conditions.$or = [
                    { _id: { $in: vectorIds } },
                    ...regexConditions
                ];
            } else {
                // If no vector matches, just Regex
                conditions.$or = regexConditions;
            }
        }



        // Find and return data based on final conditions
        let data = await food.find(conditions).lean(); // Use lean() to return headers

        // Sort and Attach Scores
        // if (search && conditions._id && conditions._id.$in && semanticScoreMap) {
        //     // Sorting by exact vector ID order is tricky with Hybrid search.
        //     // We will rely on Score attachment to let frontend sort if needed, or default Mongo sort.
        // }

        // 2. Attach Scores (We need to access scoreMap from the upper scope)
        // To do this cleanly without a massive refactor, we can re-map here if we had the map.
        // But 'scoreMap' is inside the try block.
        // Let's rely on the fact that if we sorted by ID list, we can infer relevance, but User wants NUMBERS.
        // We must move scoreMap to outer scope.
        data = data.map((item: any) => {
            const score = semanticScoreMap?.get(item._id.toString());
            return score !== undefined ? { ...item, score } : item;
        });


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

