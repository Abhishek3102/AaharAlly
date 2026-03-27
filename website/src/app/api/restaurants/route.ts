import { NextResponse } from 'next/server';
import { mongoConnect } from '@/app/utils/feature';
import { Restaurant } from '@/app/models/Restaurant';
import '@/app/models/Food'; // Register Food schema for populate('menu')

export const dynamic = 'force-dynamic';

export async function GET() {
    try {
        await mongoConnect();
        const restaurants = await Restaurant.find({}).sort({ createdAt: -1 }).populate('menu');

        // Filter out null menu items (orphaned references) for each restaurant
        restaurants.forEach((restaurant: any) => {
            if (restaurant.menu && Array.isArray(restaurant.menu)) {
                restaurant.menu = restaurant.menu.filter((item: any) => item !== null);
            }
        });

        return NextResponse.json({ success: true, restaurants });
    } catch (error: any) {
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}
