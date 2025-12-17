"use client";
import { Carousel } from "@material-tailwind/react";
import Image from "next/image";

const images = [
    '/images/food1.svg',
    '/images/food2.svg',
    '/images/food3.svg',
    '/images/food4.svg',
    '/images/food5.svg',
    '/images/food6.svg'
];

const HeroCarousel = () => {
    return (
        <div className="w-full h-[50vh] md:h-[60vh] lg:h-[70vh]">
            {/* @ts-expect-error: Material Tailwind types conflict with React 19 */}
            <Carousel
                placeholder="" // Required prop for MT v2
                className="rounded-none" // Hero doesn't need rounded corners usually
                autoplay={true}
                loop={true}
                autoplayDelay={3000} // Matches roughly what we had
                transition={{ duration: 0.5 }} // Smooth transition
                navigation={({ setActiveIndex, activeIndex, length }) => (
                    <div className="absolute bottom-4 left-2/4 z-50 flex -translate-x-2/4 gap-2">
                        {new Array(length).fill("").map((_, i) => (
                            <span
                                key={i}
                                className={`block h-1 cursor-pointer rounded-2xl transition-all content-[''] ${
                                    activeIndex === i ? "w-8 bg-white" : "w-4 bg-white/50"
                                }`}
                                onClick={() => setActiveIndex(i)}
                            />
                        ))}
                    </div>
                )}
            >
                {images.map((img, index) => (
                    <div key={index} className="relative h-full w-full">
                         <img
                            src={img}
                            alt={`Hero Food ${index + 1}`}
                            className="h-full w-full object-cover"
                        />
                    </div>
                ))}
            </Carousel>
        </div>
    );
};

export default HeroCarousel;
