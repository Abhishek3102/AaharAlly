import { NextResponse } from "next/server";

export async function POST(req: Request) {
    try {
        const { foodName, modifiers } = await req.json();

        if (!process.env.NEXT_PUBLIC_GENAI) {
            throw new Error("Missing Google AI Key (NEXT_PUBLIC_GENAI)");
        }

        // Use Direct HTTP Call to ensure we hit the Image endpoint correctly
        // Endpoint: https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict

        const apiKey = process.env.NEXT_PUBLIC_GENAI;
        const url = `https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict?key=${apiKey}`;

        const prompt = `A professional food photography shot of a delicious ${foodName}. 
    The food has the following specific modifications: ${modifiers.join(", ")}.
    The image should be hyper-realistic, 4k resolution, studio lighting, appetizing, with a soft bokeh background of a high-end restaurant table.
    Center the food on a beautiful ceramic plate.`;

        const payload = {
            instances: [
                {
                    prompt: prompt
                }
            ],
            parameters: {
                sampleCount: 1,
                aspectRatio: "1:1"
            }
        };

        console.log("🎨 Generating Image for:", prompt);

        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            const errorMessage = data.error?.message || "API Request Failed";

            // FALLBACK for Demo/Resume purposes if Billing is not enabled
            if (errorMessage.includes("billed users") || errorMessage.includes("not found")) {
                console.warn("⚠️ MOCK MODE ACTIVATED: Google Cloud Billing is not enabled. Returning demo image.");
                return NextResponse.json({
                    success: true,
                    imageUrl: "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?auto=format&fit=crop&w=1000&q=80", // High-res "AI Style" Food
                    mock: true
                });
            }

            throw new Error(errorMessage);
        }

        // Parse Response (Standard Google Cloud Prediction format)
        // Predictions[0].bytesBase64Encoded
        if (data.predictions && data.predictions[0] && data.predictions[0].bytesBase64Encoded) {
            const base64Image = data.predictions[0].bytesBase64Encoded;
            return NextResponse.json({
                success: true,
                imageUrl: `data:image/png;base64,${base64Image}`
            });
        } else {
            console.log("Unexpected API Response:", JSON.stringify(data).substring(0, 200));
            throw new Error("No image data found in response");
        }

    } catch (error: any) {
        console.error("Image Generation Failed:", error);
        return NextResponse.json({ success: false, error: error.message }, { status: 500 });
    }
}
