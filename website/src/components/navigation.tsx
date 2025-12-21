"use client";
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Search } from 'lucide-react';
import Input from '@/components/Search';
import { SignInButton, UserButton, SignedIn, SignedOut } from '@clerk/nextjs';

const Navbar = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [isScrolled, setIsScrolled] = useState(false);
    const [searchText, setSearchText] = useState("");
    const router = useRouter();

    const handleSearch = () => {
        if (searchText.trim()) {
            router.push(`/search?search=${searchText}`);
        } else {
            router.push(`/`);
        }
    };

    const toggleMenu = () => {
        setIsOpen(!isOpen);
    };

    const MenuLinks = [
        {
            id: 1,
            name: 'Explore',
            link: '/explore'
        },
        {
            id: 2,
            name: 'Cart',
            link: '/cart'
        },
        {
            id: 3,
            name: 'Favourites',
            link: '/favorites'
        },
        {
           id: 4, 
           name: 'Orders',
           link: '/orders'
        }
    ];

    const handleScroll = () => {
        setIsScrolled(window.scrollY > 0);
    };

    const handleResize = () => {
        if (window.innerWidth >= 800) {
            setIsOpen(false);
        }
    };

    const [isMounted, setIsMounted] = useState(false);

    useEffect(() => {
        setIsMounted(true);
        window.addEventListener('scroll', handleScroll);
        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('scroll', handleScroll);
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    return (
        <nav className={`fixed top-0 w-full py-2 z-50 ${isScrolled ? 'bg-white shadow-md shadow-opacity-40 shadow-black' : 'bg-white'} transition bg-transparent duration-300`}>
            <div className="container mx-auto flex justify-between items-center px-2 md:px-10 py-2">
                <div className='flex justify-start items-center gap-4'>
                    <Link href="/" className="text-lg md:text-2xl font-bold text-redCustom duration-300 hover:text-orangeCustom flex">
                        AaharAlly
                    </Link>
                    <div className="hidden md:flex items-center bg-gray-100 rounded-full px-3 py-1">
                        <input
                            type="text"
                            placeholder="Search..."
                            className="bg-transparent outline-none text-sm text-gray-700 w-32 lg:w-48 placeholder:text-gray-400"
                            value={searchText}
                            onChange={(e) => setSearchText(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        />
                        <button onClick={handleSearch} className="text-gray-500 hover:text-redCustom">
                            <Search size={18} />
                        </button>
                    </div>
                </div>
                <div className="hidden lg:flex space-x-8 items-center">
                    {MenuLinks.map((link) => (
                        <Link key={link.id} href={link.link} className="text-gray-700 hover:text-redCustom text-base font-medium transition-colors duration-200">
                            {link.name}
                        </Link>
                    ))}
                </div>
                <div className="lg:hidden">
                    <button onClick={toggleMenu} className={`text-slate-950 focus:outline-none ${!isOpen ? 'block justify-between' : 'hidden'}`}>
                        <svg
                            className={`w-6 h-6 transition-transform transform ${isOpen ? 'rotate-180' : 'rotate-0'} duration-300`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d={isOpen ? 'M6 18L18 6M6 6l12 12' : 'M4 6h16M4 12h16m-7 6h7'}
                            />
                        </svg>
                    </button>
                </div>
                <div className="flex items-center gap-2">
                    {isMounted && (
                        <>
                            <SignedIn>
                                <UserButton />
                            </SignedIn>
                            <SignedOut>
                                <SignInButton />
                            </SignedOut>
                        </>
                    )}
                </div>
            </div>
            <div
                className={`fixed inset-0 z-40 bg-black bg-opacity-50 transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                onClick={toggleMenu}
            />
            <div
                className={`fixed inset-y-0 right-0 w-full bg-white p-4 transform transition-transform duration-300 z-50 ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}
            >
                <button onClick={toggleMenu} className="text-slate-950 w-full h-8 flex items-center justify-between focus:outline-none">
                    <Link href="/" className='flex'>
                        <span className='font-bold text-redCustom text-lg px-1'>AaharAlly</span>
                    </Link>
                    <svg
                        className="w-6 h-6"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
                <div className="flex flex-col justify-items-start items-center mt-10">
                    {MenuLinks.map((link) => (
                        <Link
                            key={link.id}
                            href={link.link}
                            className="text-slate-950 text-lg sm:text-2xl duration-300 hover:bg-slate-100 p-2 rounded"
                            onClick={toggleMenu}
                        >
                            {link.name}
                        </Link>
                    ))}
                </div>
            </div>
        </nav >
    );
};

export default Navbar;
