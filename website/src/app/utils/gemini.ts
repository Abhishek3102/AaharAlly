import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GENAI! || "");

export async function filterFoodsByCondition(foods: any[], condition: string) {
    if (!condition) return foods;

    const model = genAI.getGenerativeModel({
        model: "gemini-flash-lite-latest",
    });

    // Create a simplified list for the prompt to save tokens and reduce complexity
    const foodList = foods.map((f) => ({
        id: f._id,
        name: f.name,
        description: f.description,
        ingredients: f.ingredients || "", // Assuming ingredients might be relevant if available
    }));

    const prompt = `
    You are a nutritionist assistant. I have a list of foods and a specific health condition: "${condition}".
    
    Task: Identify which of the following foods are GENERALLY considered safe or acceptable for someone with this condition.
    Strictly exclude foods that are known triggers or harmful for "${condition}".
    
    Input Foods:
    ${JSON.stringify(foodList)}
    
    Output Format:
    Return ONLY a raw JSON array of strings, where each string is the "id" of a safe food item. 
    Do not include markdown formatting (like \`\`\`json), just the array.
    Example: ["id1", "id2", "id5"]
  `;

    try {
        const result = await model.generateContent(prompt);
        const responseText = result.response.text();

        // Clean up response if it contains markdown code blocks
        const cleanText = responseText.replace(/```json/g, "").replace(/```/g, "").trim();

        const safeFoodIds = JSON.parse(cleanText);

        if (!Array.isArray(safeFoodIds)) {
            console.error("Gemini did not return an array", safeFoodIds);
            return foods; // Fallback to all foods or empty? Safer to return all and let user decide or empty? 
            // Let's return foods but log error. Actually, if it fails, maybe return original to not break UI, but for health filtering, maybe empty is safer?
            // Let's return original with a warning for now to avoid empty screens on AI hiccups.
            return foods;
        }

        // Filter the original list
        const filteredFoods = foods.filter(f => safeFoodIds.includes(f._id.toString()));
        return filteredFoods;

    } catch (error) {
        console.error("Error filtering foods with Gemini:", error);
        return foods; // Fallback to returning all foods if AI fails
    }
}
