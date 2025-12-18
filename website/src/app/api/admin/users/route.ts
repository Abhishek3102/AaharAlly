import { NextResponse } from 'next/server';
import { User } from '@/app/models/User';
import { Order } from '@/app/models/Order';
import { mongoConnect } from '@/app/utils/feature';
import jwt from 'jsonwebtoken';
import { cookies } from 'next/headers';

export async function GET() {
    try {
        await mongoConnect();

        // Auth Check
        const cookieStore = await cookies();
        const token = cookieStore.get('admin_token')?.value;

        if (!token) {
            return NextResponse.json({ success: false, message: 'Unauthorized' }, { status: 401 });
        }

        try {
            jwt.verify(token, process.env.JWT_SECRET || 'fallback_secret_key_change_me');
        } catch (err) {
            return NextResponse.json({ success: false, message: 'Invalid Token' }, { status: 401 });
        }

        // Fetch Users with Order Counts
        // Since we don't have a direct relation in schema for virtual populates easily set up, 
        // we'll fetch users and aggregate orders.
        // Ideally, we'd use an aggregation pipeline on User.

        // Simple approach for now (assumes < 10k users, else pagination needed):
        const users = await User.find({}).sort({ createdAt: -1 }).lean();

        const userStats = await Promise.all(users.map(async (user: any) => {
            const orderCount = await Order.countDocuments({ userId: user._id });

            // Calculate "Joined" duration or date
            const joinDate = user.createdAt || user._id.getTimestamp(); // Fallback to ObjectId timestamp if createdAt missing

            return {
                _id: user._id,
                email: user.email,
                joinedAt: joinDate,
                orderCount: orderCount,
            };
        }));

        return NextResponse.json({ success: true, users: userStats }, { status: 200 });

    } catch (error: any) {
        console.error('Admin Users API Error:', error);
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}
