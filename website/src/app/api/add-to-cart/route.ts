import { User } from "../../models/User";
import { mongoConnect } from "../../utils/feature";
import { NextResponse } from "next/server";

export async function POST(req: Request) {
    try {
        await mongoConnect();
        const { email, foodId } = await req.json();

        if (!email || !foodId) {
            return NextResponse.json({ success: false, message: "Missing fields" }, { status: 400 });
        }

        const user = await User.findOne({ email });

        if (!user) {
            return NextResponse.json({ success: false, message: "User not found" }, { status: 404 });
        }

        if (!user.cart) {
            user.cart = [];
        }

        const itemIndex = user.cart.findIndex(
            (item: { foodId: { toString: () => string }; quantity: number }) => item.foodId.toString() === foodId
        );

        if (itemIndex > -1) {
            user.cart[itemIndex].quantity += 1;
        } else {
            user.cart.push({ foodId, quantity: 1 });
        }

        await user.save();

        return NextResponse.json({ success: true, message: "Item added to cart", cart: user.cart }, { status: 200 });
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
        console.error("Add to cart error:", error);
        return NextResponse.json({ success: false, message: error.message || "Error adding to cart" }, { status: 500 });
    }
}
