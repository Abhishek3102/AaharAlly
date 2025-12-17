"use client";
import { Carousel } from "@material-tailwind/react";
import Image from "next/image";

export function BannerCarousel() {

  const images = [
    "/images/food1.svg",
    "/images/food2.svg",
    "/images/food3.svg",
    "/images/food4.svg",
    "/images/food5.svg",
    "/images/food6.svg",
  ];

  return (
    // @ts-expect-error: Material Tailwind types conflict with React 19
    <Carousel
      placeholder={"HomeCarousel"}
      className="rounded-3xl h-[40vh]"
      autoplay={true}
      loop={true}
      navigation={({ setActiveIndex, activeIndex, length }) => (
        <div className="absolute bottom-4 left-2/4 z-50 hidden -translate-x-2/4 gap-2">
          {new Array(length).fill("").map((_, i) => (
            <span
              key={i}
              className={`block h-1 cursor-pointer rounded-3xl transition-all content-[''] ${
                activeIndex === i ? "w-8 bg-white" : "w-4 bg-white/50"
              }`}
              onClick={() => setActiveIndex(i)}
            />
          ))}
        </div>
      )} 
      >
        {images.map((image, index) => (
          <Image
            key={index}
            src={image}
            alt={`image ${index}`}
            className="h-full w-full object-cover"
            width={500}
            height={500}
            priority={index === 0}
          />
        ))}
    </Carousel>
  );
}
