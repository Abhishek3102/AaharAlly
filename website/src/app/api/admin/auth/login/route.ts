import { NextResponse } from 'next/server';
import { Admin } from '@/app/models/Admin';
import { mongoConnect } from '@/app/utils/feature';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

export async function POST(req: Request) {
    try {
        await mongoConnect();
        const { email, password } = await req.json();

        if (!email || !password) {
            return NextResponse.json({ success: false, message: 'Missing credentials' }, { status: 400 });
        }

        const admin = await Admin.findOne({ email });

        if (!admin) {
            return NextResponse.json({ success: false, message: 'Invalid credentials' }, { status: 401 });
        }

        const isMatch = await bcrypt.compare(password, admin.password);

        if (!isMatch) {
            return NextResponse.json({ success: false, message: 'Invalid credentials' }, { status: 401 });
        }

        const token = jwt.sign(
            { id: admin._id, email: admin.email, role: 'admin' },
            process.env.JWT_SECRET || 'fallback_secret_key_change_me',
            { expiresIn: '1d' }
        );

        const response = NextResponse.json({ success: true, message: 'Login successful' }, { status: 200 });

        // Set secure cookie
        response.cookies.set('admin_token', token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'strict',
            maxAge: 60 * 60 * 24, // 1 day
            path: '/',
        });

        return response;

    } catch (error: any) {
        console.error('Admin Login Error:', error);
        return NextResponse.json({ success: false, message: error.message }, { status: 500 });
    }
}
