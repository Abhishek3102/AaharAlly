
import { currentUser } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { mongoConnect } from "@/app/utils/feature";
import { User } from "@/app/models/User";
import { Order } from "@/app/models/Order";
import { food } from "@/app/models/Food";
import axios from "axios";

export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
    try {
        const user = await currentUser();
        if (!user) {
            return NextResponse.json({ success: false, message: "Unauthorized" }, { status: 401 });
        }

        await mongoConnect();

        const email = user.emailAddresses[0].emailAddress;
        const dbUser = await User.findOne({ email }).populate("cart.foodId");

        if (!dbUser || dbUser.cart.length === 0) {
            return NextResponse.json({ success: false, message: "Cart is empty" }, { status: 400 });
        }

        const { shippingInfo } = await req.json();

        // Calculate Total
        let totalAmount = 0;
        const orderItems = dbUser.cart.map((item: any) => {
            const price = parseFloat(item.foodId.price.replace(/[^0-9.]/g, '')); // Handle "₹ 200" or similar
            const total = price * item.quantity;
            totalAmount += total;
            return {
                foodId: item.foodId._id,
                quantity: item.quantity,
                price: price,
                // Meta for ML
                category: item.foodId.category,
                meal_type: item.foodId.meal_type
            };
        });

        // Create Order
        const newOrder = await Order.create({
            userId: dbUser._id,
            items: orderItems.map(({ foodId, quantity, price }: any) => ({ foodId, quantity, price })),
            totalAmount,
            shippingInfo, // Ensure Order model has this if needed, or just store minimal
            status: 'completed' // Assuming direct success for now as payment gate is dummy
        });

        // Clear Cart & Update Last Order Date
        dbUser.cart = [];
        dbUser.lastOrderDate = new Date();
        // Invalidate recommendations cache explicitly?
        // dbUser.recommendedCategories = []; // Optional, or let time/date check handle it
        await dbUser.save();

        // Background: Notify ML Service
        // We do this asynchronously (fire and forget) to keep UI fast
        // or await if we want to be sure. Await is safer for now.
        const ML_API = process.env.NEXT_PUBLIC_ML_API_URL || "https://aaharally.onrender.com";

        // We notify for EACH item or unique category?
        // ML expects single order entries usually.
        // Let's iterate unique categories bought.
        const uniqueItems = orderItems;

        // We might need to batch this or just loop.
        for (const item of uniqueItems) {
            try {
                await axios.post(`${ML_API}/api/store_order`, {
                    user_id: user.id, // Clerk ID
                    restaurant_id: "AaharAlly_Main",
                    meal_category: item.category, // e.g. "Create your Own" or "North Indian"
                    age: dbUser.age,
                    gender: dbUser.gender,
                    review: "" // No reviewer yet
                });
            } catch (err) {
                console.error("Failed to sync order to ML:", err);
            }
        }

        return NextResponse.json({ success: true, orderId: newOrder._id });

    } catch (error: any) {
        console.error("Place Order Error:", error);
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}

export async function GET(req: Request) {
    try {
        const user = await currentUser();
        if (!user) {
            return NextResponse.json({ success: false, message: "Unauthorized" }, { status: 401 });
        }
        await mongoConnect();
        const email = user.emailAddresses[0].emailAddress;
        const dbUser = await User.findOne({ email });

        if (!dbUser) return NextResponse.json({ success: false }, { status: 404 });

        const orders = await Order.find({ userId: dbUser._id })
            .sort({ createdAt: -1 })
            .populate("items.foodId");

        return NextResponse.json({ success: true, orders });
    } catch (error: any) {
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}
