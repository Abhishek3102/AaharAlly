"use client"
import { SignIn, SignedOut, SignedIn, useUser } from '@clerk/nextjs'
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

const RedirectToHome = () => {
    const router = useRouter();
    useEffect(() => {
        router.push('/');
    }, [router]);
    return <div className="flex h-screen items-center justify-center"><p>Redirecting...</p></div>;
};

export default function Page() {
    const [wHeight, setWHeight] = useState(0);

    useEffect(() => {
        setWHeight(window?.innerHeight * 0.89);
    }, []);

    return (
        <>
            <SignedIn>
                <RedirectToHome />
            </SignedIn>
            <SignedOut>
                <span className='flex justify-center items-center mt-16' style={{ height: wHeight }}>
                    <SignIn />
                </span>
            </SignedOut>
        </>
    );
}
