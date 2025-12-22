
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

        // Filter out null foodIds (Orphans)
        const validCartItems = dbUser.cart.filter((item: any) => item.foodId);

        if (validCartItems.length === 0) {
            return NextResponse.json({ success: false, message: "Cart items are unavailable" }, { status: 400 });
        }

        // Calculate Total
        let totalAmount = 0;
        const orderItems = validCartItems.map((item: any) => {
            // Fix: Handle ranges like "$12-16"
            let priceStr = item.foodId.price || "0";
            if (priceStr.includes('-')) {
                priceStr = priceStr.split('-')[0];
            }
            const price = parseFloat(priceStr.replace(/[^0-9.]/g, ''));

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

        // Background: Notify ML Service (Optimized)
        // We use Promise.allSettled with a strict timeout to ensure this never hangs the UI > 2 seconds
        const ML_API = process.env.NEXT_PUBLIC_ML_API_URL || "https://aaharally.onrender.com";
        const uniqueItems = orderItems;

        const mlPromises = uniqueItems.map((item: any) => {
            return axios.post(`${ML_API}/api/store_order`, {
                user_id: user.id, // Clerk ID
                restaurant_id: "AaharAlly_Main",
                meal_category: item.category,
                age: dbUser.age,
                gender: dbUser.gender,
                review: ""
            }, { timeout: 2500 }); // 2.5s Timeout
        });

        // Wait for all, but don't fail if they fail.
        await Promise.allSettled(mlPromises);

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
