
import { NextResponse } from "next/server";

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET() {
    try {
        const { mongoConnect } = await import("@/app/utils/feature");
        const { food } = await import("@/app/models/Food");
        await mongoConnect();

        const categories = await food.distinct("category");
        return NextResponse.json({ success: true, categories });
    } catch (e: any) {
        return NextResponse.json({ error: e.message });
    }
}
