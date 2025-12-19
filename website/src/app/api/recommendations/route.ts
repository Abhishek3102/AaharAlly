
import { currentUser } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
// Use require for feature if import is suspect, but import should work.
import { mongoConnect } from "@/app/utils/feature";
import { User } from "@/app/models/User";
import { food } from "@/app/models/Food";
import axios from "axios";

export const dynamic = 'force-dynamic';

export async function GET(req: Request) {
    let stage = "init";
    try {
        stage = "auth_check";
        const user = await currentUser();
        if (!user) {
            return NextResponse.json({ success: false, message: "Unauthorized", stage }, { status: 401 });
        }

        stage = "mongo_connect";
        try {
            await mongoConnect();
        } catch (dbErr: any) {
            console.error("DB Connection Error:", dbErr);
            throw new Error(`DB Connection Failed: ${dbErr.message}`);
        }

        stage = "user_lookup";
        const email = user.emailAddresses?.[0]?.emailAddress;
        if (!email) {
            return NextResponse.json({ success: false, message: "No email found for user", stage }, { status: 400 });
        }

        const dbUser = await User.findOne({ email });

        if (!dbUser || dbUser.age === undefined || !dbUser.gender) {
            return NextResponse.json({
                success: false,
                message: "Profile incomplete",
                action: "complete_profile",
                stage
            }, { status: 200 });
        }

        // 2. CHECK CACHE
        const ONE_DAY = 24 * 60 * 60 * 1000;
        const now = new Date();
        const lastRec = dbUser.lastRecommendationDate ? new Date(dbUser.lastRecommendationDate).getTime() : 0;
        const lastOrder = dbUser.lastOrderDate ? new Date(dbUser.lastOrderDate).getTime() : 0;

        let recommendedCategories: string[] = [];
        let fromCache = false;
        let mlMeta: any = {};

        // Valid if: has categories AND (fresh enough OR no recent order since last rec)
        // Logic: Refresh if older than 24h OR if user ordered something *after* the last recommendation
        const isCacheValid = dbUser.recommendedCategories && dbUser.recommendedCategories.length > 0 &&
            (now.getTime() - lastRec < ONE_DAY) &&
            (lastRec > lastOrder);

        if (isCacheValid) {
            console.log("Serving recommendations from CACHE");
            recommendedCategories = dbUser.recommendedCategories!;
            fromCache = true;
            mlMeta = { user_type: dbUser.user_type, cluster: dbUser.cluster };
        } else {
            // 3. CALL ML SERVICE (Refresh)
            stage = "ml_call";
            const ML_API = process.env.NEXT_PUBLIC_ML_API_URL || "https://aaharally.onrender.com";
            // Use Clerk ID (user.id) as the unique ML user_id
            const payload = {
                user_id: user.id,
                age: dbUser.age,
                gender: dbUser.gender,
                restaurant_id: "AaharAlly_Main"
            };

            console.log("Calling ML API (Cache Miss/Stale):", payload);

            let mlResponse;
            try {
                const res = await axios.post(`${ML_API}/api/recommend`, payload);
                mlResponse = res.data;
            } catch (mlError: any) {
                console.error("ML Service Error:", mlError.message);
                // On failure, if we have OLD cache, serve it instead of failing!
                if (dbUser.recommendedCategories && dbUser.recommendedCategories.length > 0) {
                    console.log("ML Failed, falling back to STALE CACHE");
                    recommendedCategories = dbUser.recommendedCategories;
                    fromCache = true;
                    // Don't return error, proceed to fetch food
                } else {
                    return NextResponse.json({ success: false, message: "Recommendation engine unavailable", error: mlError.message, stage }, { status: 200 });
                }
            }

            if (mlResponse && !mlResponse.success) {
                // Fallback to cache if available
                if (dbUser.recommendedCategories && dbUser.recommendedCategories.length > 0) {
                    recommendedCategories = dbUser.recommendedCategories;
                } else {
                    return NextResponse.json({ success: false, message: "ML Error: " + mlResponse.error, stage }, { status: 500 });
                }
            } else if (mlResponse && mlResponse.success) {
                recommendedCategories = mlResponse.recommendations || [];
                mlMeta = { user_type: mlResponse.user_type, cluster: mlResponse.cluster };

                // UPDATE USER CACHE
                if (recommendedCategories.length > 0) {
                    await User.updateOne({ email }, {
                        $set: {
                            recommendedCategories,
                            lastRecommendationDate: new Date(),
                            user_type: mlResponse.user_type,
                            cluster: mlResponse.cluster
                        }
                    });
                }
            }
        }

        stage = "food_fetch";

        if (recommendedCategories.length === 0) {
            return NextResponse.json({ success: true, data: [] });
        }

        const foodItems = await food.find({
            category: { $in: recommendedCategories }
        }).lean();

        stage = "sorting";
        const sortedItems: any[] = [];
        const itemsByCategory: Record<string, any[]> = {};

        foodItems.forEach((item: any) => {
            if (!itemsByCategory[item.category]) itemsByCategory[item.category] = [];
            itemsByCategory[item.category].push(item);
        });

        recommendedCategories.forEach(cat => {
            if (itemsByCategory[cat]) {
                sortedItems.push(...itemsByCategory[cat].slice(0, 4));
            }
        });

        return NextResponse.json({
            success: true,
            data: sortedItems,
            meta: {
                ...mlMeta,
                categories: recommendedCategories,
                fromCache
            }
        });

    } catch (error: any) {
        console.error(`Recommendation API Error at stage ${stage}:`, error);
        return NextResponse.json({
            success: false,
            message: `Error at ${stage}: ${error.message}`,
            stack: error.stack
        }, { status: 200 }); // Debugging: Return 200 so frontend can read the error message
    }
}

