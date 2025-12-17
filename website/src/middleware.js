import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';
import { NextResponse } from 'next/server';

const isProtectedRoute = createRouteMatcher([
    '/(.*)',
]);

const isPublicRoute = createRouteMatcher([
    '/sign-in(.*)',
    '/sign-up(.*)',
    '/item-details(.*)',
    '/',
    '/hotelUser(.*)',
    '/api/(.*)',
    '/explore(.*)',
    '/favorites(.*)',
    '/checkout(.*)',
    '/cart(.*)',
]);

export default clerkMiddleware((auth, req) => {
    const { userId } = auth();
    const currentUrl = new URL(req.url);
    const isAccessingAuthRoute = currentUrl.pathname.startsWith('/sign-in') || currentUrl.pathname.startsWith('/sign-up');

    // If user is logged in and trying to access sign-in/sign-up, redirect to home
    if (userId && isAccessingAuthRoute) {
        return NextResponse.redirect(new URL('/', req.url));
    }

    // If user is NOT logged in and trying to access a protected route, redirect to sign-in
    if (!userId && !isPublicRoute(req)) {
        return auth().protect();
    }
});

export const config = {
    matcher: [
        '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
        '/(api|trpc)(.*)',
    ],
};
