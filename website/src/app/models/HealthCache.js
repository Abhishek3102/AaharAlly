import mongoose from "mongoose";

const HealthCacheSchema = new mongoose.Schema({
    condition: {
        type: String,
        required: true,
        unique: true
    },
    safe_food_ids: [{
        type: String, // Storing as String IDs is safer for generic checking, or ObjectId
        required: true
    }],
    last_updated: {
        type: Date,
        default: Date.now
    }
});

// Use useDb to switch to the ML database specifically for this model
export const HealthCache = (mongoose.connection.useDb('aahar_ally_ml').models.HealthCache) || mongoose.connection.useDb('aahar_ally_ml').model("HealthCache", HealthCacheSchema);
