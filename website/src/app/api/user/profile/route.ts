import { NextResponse } from "next/server";
import { mongoConnect } from "@/app/utils/feature";
import { User } from "@/app/models/User";
import { currentUser } from "@clerk/nextjs/server";

export async function PATCH(req: Request) {
    try {
        await mongoConnect();
        const user = await currentUser();

        if (!user) {
            return NextResponse.json({ success: false, message: "Unauthorized" }, { status: 401 });
        }

        const email = user.emailAddresses[0].emailAddress;
        const body = await req.json();
        const { age, gender } = body;

        if (!age || !gender) {
            return NextResponse.json({ success: false, message: "Missing fields" }, { status: 400 });
        }

        const updatedUser = await User.findOneAndUpdate(
            { email },
            { $set: { age, gender } },
            { new: true, upsert: true } // Create if doesn't exist (though login should have created it)
        );

        return NextResponse.json({ success: true, user: updatedUser }, { status: 200 });

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
        console.error("Profile update error:", error);
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}

export async function GET() {
    try {
        await mongoConnect();
        const user = await currentUser();

        if (!user) {
            return NextResponse.json({ success: false, message: "Unauthorized" }, { status: 401 });
        }

        const email = user.emailAddresses[0].emailAddress;
        const dbUser = await User.findOne({ email });

        if (!dbUser) {
            return NextResponse.json({ success: false, message: "User not found" }, { status: 404 });
        }

        return NextResponse.json({
            success: true,
            profileCompleted: !!(dbUser.age && dbUser.gender),
            user: dbUser
        }, { status: 200 });

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}
