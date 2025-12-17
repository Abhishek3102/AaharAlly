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

        user.cart = user.cart.filter((item: any) => item.foodId.toString() !== foodId);

        await user.save();

        return NextResponse.json({ success: true, message: "Item removed from cart", cart: user.cart }, { status: 200 });
    } catch (error) {
        return NextResponse.json({ success: false, message: "Error removing item" }, { status: 500 });
    }
}
