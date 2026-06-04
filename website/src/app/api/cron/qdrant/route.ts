import { NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";

export async function GET(req: Request) {
    try {
        // Vercel Cron authorization (optional but recommended in production)
        // You can check req.headers.get('Authorization') against a VERCEL_CRON_SECRET if needed.

        const QDRANT_URL = process.env.QDRANT_URL;
        const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

        if (!QDRANT_URL || !QDRANT_API_KEY) {
            return NextResponse.json({ message: "Qdrant credentials missing" }, { status: 500 });
        }

        const qdrant = new QdrantClient({ url: QDRANT_URL, apiKey: QDRANT_API_KEY });
        
        // A simple query to keep the cluster active
        const collections = await qdrant.getCollections();

        console.log("Qdrant Keep-Alive Ping Successful. Collections found:", collections.collections.length);

        return NextResponse.json({ 
            success: true, 
            message: "Qdrant cluster pinged successfully.",
            collections: collections.collections.length
        });
        
    } catch (error: any) {
        console.error("Qdrant Keep-Alive Error:", error);
        return NextResponse.json({ success: false, error: error.message }, { status: 500 });
    }
}
