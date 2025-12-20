
import { NextResponse } from "next/server";
import axios from "axios";

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(req: Request) {
    let logs: string[] = [];
    try {
        const { mongoConnect } = await import("@/app/utils/feature");
        const { User } = await import("@/app/models/User");

        await mongoConnect();

        // Lean query
        const users = await User.find({}).select("email age gender user_type cluster").lean();
        logs.push(`Found ${users.length} users (V2 Fresh).`);

        const ML_API = process.env.NEXT_PUBLIC_ML_API_URL || "https://aaharally.onrender.com";
        let successCount = 0;

        for (const user of users) {
            const u = user as any;

            if (!u.age || !u.gender) {
                logs.push(`SKIPPING ${u.email}: No Age/Gender`);
                continue;
            }

            const payload = {
                user_id: u._id.toString(),
                age: u.age,
                gender: u.gender,
                restaurant_id: "AaharAlly_Main"
            };

            try {
                const res = await axios.post(`${ML_API}/api/recommend`, payload, { timeout: 15000 });

                if (res.data && res.data.success) {
                    const recommendations = res.data.recommendations || [];
                    if (recommendations.length > 0) {
                        await User.updateOne({ _id: u._id }, {
                            $set: {
                                recommendedCategories: recommendations,
                                lastRecommendationDate: new Date(),
                                user_type: res.data.user_type,
                                cluster: res.data.cluster
                            }
                        });
                        successCount++;
                        logs.push(`UPDATED ${u.email}: [${recommendations.join(', ')}]`);
                    } else {
                        logs.push(`ML EMPTY for ${u.email}. Checking defaults...`);
                    }
                } else {
                    logs.push(`ML FAIL for ${u.email}: ${res.data.error}`);

                    // FALLBACK LOGIC
                    if (res.data.error && (typeof res.data.error === 'string') && res.data.error.includes("Clustering model")) {
                        const fallbackRecs = ["Indian Curry", "Street Food", "Snacks"];
                        await User.updateOne({ _id: u._id }, {
                            $set: {
                                recommendedCategories: fallbackRecs,
                                lastRecommendationDate: new Date()
                            }
                        });
                        logs.push(`APPLIED FALLBACK for ${u.email}: [${fallbackRecs.join(', ')}]`);
                        successCount++;
                    }
                }
            } catch (mlErr: any) {
                logs.push(`ML ERROR for ${u.email}: ${mlErr.message}`);
            }
        }

        return NextResponse.json({
            success: true,
            message: `Updated ${successCount}/${users.length} users.`,
            logs
        });

    } catch (error: any) {
        return NextResponse.json({ success: false, error: error.message, logs }, { status: 500 });
    }
}
