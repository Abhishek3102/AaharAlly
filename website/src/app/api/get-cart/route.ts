import { User } from "../../models/User";
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { food } from "../../models/Food"; // Ensure Food model is registered
import { mongoConnect } from "../../utils/feature";
import { NextResponse } from "next/server";

export async function GET(req: Request) {
    try {
        await mongoConnect();
        const url = new URL(req.url);
        const email = url.searchParams.get("email");

        if (!email) {
            return NextResponse.json({ success: false, message: "Email required" }, { status: 400 });
        }

        // Ensure 'food' model is initialized before population if not imported elsewhere
        // But logically importing it is enough.

        const user = await User.findOne({ email }).populate("cart.foodId");

        if (!user) {
            return NextResponse.json({ success: false, message: "User not found" }, { status: 404 });
        }

        return NextResponse.json({ success: true, cart: user.cart }, { status: 200 });
    } catch (error) {
        console.error(error);
        return NextResponse.json({ success: false, message: "Error fetching cart" }, { status: 500 });
    }
}
