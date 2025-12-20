
import { currentUser } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
// Dynamic imports for safety
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(req: Request) {
    let stage = "init";
    try {
        stage = "imports";
        const { mongoConnect } = await import("@/app/utils/feature");
        const { User } = await import("@/app/models/User");
        const { food } = await import("@/app/models/Food");

        stage = "auth_check";
        const user = await currentUser();
        if (!user) {
            return NextResponse.json({ success: false, message: "Unauthorized" }, { status: 401 });
        }

        stage = "mongo_connect";
        await mongoConnect();

        const email = user.emailAddresses?.[0]?.emailAddress;
        if (!email) return NextResponse.json({ success: false, message: "No email" }, { status: 400 });

        const dbUser = await User.findOne({ email });

        // If user not found or profile incomplete
        if (!dbUser || !dbUser.age || !dbUser.gender) {
            return NextResponse.json({ success: false, action: "complete_profile" });
        }

        // 1. READ ONLY - NO ML CALLS
        // We simply trust whatever is in the DB. 
        // If nothing is in DB, we fallback to NULL (Frontend handles empty state or we send popular items).

        let recommendedCategories = dbUser.recommendedCategories || [];

        // Fallback: If absolutely no recommendations exist yet, maybe suggest "North Indian" or Random?
        // For now, let's return generic popular categories if empty
        if (recommendedCategories.length === 0) {
            // Fallback to valid DB categories
            recommendedCategories = ["Indian Curry", "Street Food", "Snacks"];
        }

        const foodItems = await food.find({
            category: { $in: recommendedCategories }
        }).lean();

        // Sort items by category bucket
        const sortedItems: any[] = [];
        const itemsByCategory: Record<string, any[]> = {};

        foodItems.forEach((item: any) => {
            if (!itemsByCategory[item.category]) itemsByCategory[item.category] = [];
            itemsByCategory[item.category].push(item);
        });

        recommendedCategories.forEach((cat: string) => {
            if (itemsByCategory[cat]) {
                sortedItems.push(...itemsByCategory[cat].slice(0, 4));
            }
        });

        return NextResponse.json({
            success: true,
            data: sortedItems,
            meta: {
                categories: recommendedCategories,
                source: "database_cache"
            }
        });

    } catch (error: any) {
        console.error(`Rec API Error at ${stage}:`, error);
        return NextResponse.json({ success: false, message: error.message }, { status: 200 });
    }
}
