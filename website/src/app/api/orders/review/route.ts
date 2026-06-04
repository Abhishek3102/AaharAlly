
import { currentUser } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { mongoConnect } from "@/app/utils/feature";
import { User } from "@/app/models/User";
import { food } from "@/app/models/Food";
import axios from "axios";

export async function POST(req: Request) {
    try {
        const user = await currentUser();
        if (!user) {
            return NextResponse.json({ success: false, message: "Unauthorized" }, { status: 401 });
        }

        await mongoConnect();

        const { foodId, reviewText } = await req.json();

        if (!foodId || !reviewText) {
            return NextResponse.json({ success: false, message: "Missing foodId or reviewText" }, { status: 400 });
        }

        const email = user.emailAddresses[0].emailAddress;
        const dbUser = await User.findOne({ email });

        if (!dbUser) {
            return NextResponse.json({ success: false, message: "User not found" }, { status: 404 });
        }

        // 1. Get the category of the food being reviewed
        const foodItem = await food.findById(foodId);
        if (!foodItem) {
            return NextResponse.json({ success: false, message: "Food item not found" }, { status: 404 });
        }
        const category = foodItem.category;

        // 2. Call ML Service for Sentiment Analysis
        const ML_API = process.env.NEXT_PUBLIC_ML_API_URL || "https://aaharally.onrender.com";
        let isPositive = true;
        let mlSuccess = false;

        try {
            const mlRes = await axios.post(`${ML_API}/api/sentiment/predict`, {
                texts: [reviewText]
            }, { timeout: 3000 });

            if (mlRes.data.success && mlRes.data.positive_probabilities) {
                const prob = mlRes.data.positive_probabilities[0];
                isPositive = prob >= 0.5;
                mlSuccess = true;
                console.log(`Sentiment Result: ${isPositive ? 'Pos' : 'Neg'} (Prob: ${prob}) for text: "${reviewText}"`);
            }
        } catch (e) {
            console.error("ML Sentiment Analysis HTTP Call Failed:", e.message);
        }

        // FALLBACK: If ML service is down, sleeping, or models aren't trained
        if (!mlSuccess) {
            console.log("Using Fallback Keyword Sentiment Analysis...");
            const textLower = reviewText.toLowerCase();
            
            // Strong negative indicators
            const negativeWords = ["not good", "bad", "terrible", "awful", "worst", "disappointed", "hate", "gross", "poor", "never", "bland", "cold"];
            
            // Check if any negative word is in the text
            const hasNegative = negativeWords.some(word => textLower.includes(word));
            
            if (hasNegative) {
                isPositive = false;
                console.log(`Fallback Sentiment Result: Negative (Matched negative keywords in: "${reviewText}")`);
            } else {
                isPositive = true;
                console.log(`Fallback Sentiment Result: Positive (Default/No negative keywords)`);
            }
        }

        // 3. Update User Recommendations based on Sentiment (Re-ranking)
        let updatedCategories = dbUser.recommendedCategories || [];

        if (isPositive) {
            // Add category if not present (Boost)
            if (!updatedCategories.includes(category)) {
                updatedCategories.push(category);
                // Keep the list manageable
                if (updatedCategories.length > 6) updatedCategories.shift();
            }
        } else {
            // Remove category if present (Penalize)
            updatedCategories = updatedCategories.filter((cat: string) => cat !== category);
        }

        dbUser.recommendedCategories = updatedCategories;
        await dbUser.save();

        return NextResponse.json({ 
            success: true, 
            message: `Review submitted. Sentiment: ${isPositive ? 'Positive' : 'Negative'}. Preferences updated.`,
            isPositive
        });

    } catch (error: any) {
        console.error("Review API Error:", error);
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}
