import { User } from "../../models/User";
import { mongoConnect } from "../../utils/feature";
import { NextRequest, NextResponse } from "next/server";
import { food } from "../../models/Food"; // Register 'food' model

mongoConnect();

export async function GET(req: NextRequest) {
  const email = req.nextUrl.searchParams.get("email");
  try {
    const user = await User.findOne({ email: email }).populate("favoriteFoods");
    if (!user) {
      return NextResponse.json({ message: "User not found" }, { status: 404 });
    }

    return NextResponse.json(
      { favoriteFoods: user.favoriteFoods },
      { status: 200 }
    );
  } catch (error) {
    console.log(error);
    return NextResponse.json(
      { message: "Internal server error", error },
      { status: 500 }
    );
  }
}
