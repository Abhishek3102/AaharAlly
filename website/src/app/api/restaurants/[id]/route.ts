import { NextResponse } from 'next/server';
import { mongoConnect } from '@/app/utils/feature';
import { Restaurant } from '@/app/models/Restaurant';
import { food } from '@/app/models/Food'; // Import to ensure schema registration

export async function GET(req: Request, context: { params: Promise<{ id: string }> }) {
    try {
        await mongoConnect();
        const { id } = await context.params;

        // Ensure Food model is registered before population
        // (Just referencing the import is usually enough, but using it safely is better)
        const _ = food;

        const restaurant = await Restaurant.findById(id).populate('menu');

        if (!restaurant) {
            return NextResponse.json({ success: false, message: "Restaurant not found" }, { status: 404 });
        }

        // Filter out any null menu items (orphaned references)
        restaurant.menu = restaurant.menu.filter((item: any) => item !== null);

        return NextResponse.json({ success: true, restaurant });
    } catch (error: any) {
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}
