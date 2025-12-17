import { userData } from "@/app/models/UserData";
import { food } from "../../models/Food";
import FoodPreference from "../../models/FoodPreferenceSchema";
import { mongoConnect } from "../../utils/feature";
import { NextResponse } from "next/server";

export async function GET(req) {
    const url = new URL(req.url);
    const ageParam = url.searchParams.get('age');
    const id = url.searchParams.get('id');
    const regions = url.searchParams.get('regions'); // Allow multiple regions
    const categoryParam = url.searchParams.get('categories'); // Allow multiple categories
    const meal_type = url.searchParams.get('meal_type');
    const search = url.searchParams.get('search');

    try {
        await mongoConnect();

        // Find specific item by ID
        if (id) {
            const data = await food.findById({ _id: id });
            return NextResponse.json({ data, success: true }, { status: 200 });
        }

        const conditions = {};
        let categoriesArray = [];
        let regionsArray = [];

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
                    const mostConsumedCategory = data.categories.reduce((prev, current) => {
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
        if (search) {
            const searchRegex = new RegExp(search, 'i');
            conditions.$or = [
                { name: searchRegex },
                { description: searchRegex },
                { category: searchRegex }
            ];
        }

        // Find and return data based on final conditions
        const data = await food.find(conditions);
        return NextResponse.json({ data, success: true }, { status: 200 });
    } catch (err) {
        return NextResponse.json({ message: `Error processing request: ${err.message}`, success: false }, { status: 500 });
    }
}
