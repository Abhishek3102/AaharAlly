import { NextResponse } from 'next/server';
import { mongoConnect } from '@/app/utils/feature';
import { Restaurant } from '@/app/models/Restaurant';

export const dynamic = 'force-dynamic';

export async function GET() {
    try {
        await mongoConnect();
        const restaurants = await Restaurant.find({}).sort({ createdAt: -1 });
        return NextResponse.json({ success: true, restaurants });
    } catch (error: any) {
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}
