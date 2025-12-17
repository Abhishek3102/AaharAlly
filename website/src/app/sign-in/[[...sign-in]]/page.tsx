"use client"
import { SignIn } from '@clerk/nextjs'
import { useEffect, useState } from 'react';

// Custom function to handle Google login
const handleGoogleLogin = async () => {
    const response = await fetch('/api/auth/google', { method: 'POST' });
    const { url } = await response.json();
    window.location.href = url; // Redirect to Google's OAuth page
};

export default function Page() {
    const [wHeight, setWHeight] = useState(0);

    useEffect(() => {
        setWHeight(window?.innerHeight * 0.89);
    }, []);

    return (
        <>
            <span className='flex justify-center items-center mt-16' style={{ height: wHeight }}>
                <SignIn />
            </span>
        </>
    );
}
