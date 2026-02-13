import mongoose, { Schema, Document, Model } from 'mongoose';

export interface IRestaurant extends Document {
    name: string;
    description: string;
    address: string;
    location: {
        lat: number;
        lng: number;
    };
    image: string; // Hotel Cover Image
    owner: {
        name: string;
        image: string; // Owner Profile Pick
    };
    rating: number;
    menu: string[]; // Array of Food IDs
    createdAt: Date;
    updatedAt: Date;
}

const RestaurantSchema: Schema<IRestaurant> = new mongoose.Schema({
    name: { type: String, required: true },
    description: { type: String, required: true },
    address: { type: String, required: true },
    location: {
        lat: { type: Number, required: true },
        lng: { type: Number, required: true }
    },
    image: { type: String, required: true },
    owner: {
        name: { type: String, required: true },
        image: { type: String, required: true }
    },
    rating: { type: Number, default: 4.5 },
    menu: [{ type: Schema.Types.ObjectId, ref: 'food' }],
}, { timestamps: true });

export const Restaurant: Model<IRestaurant> = mongoose.models.Restaurant || mongoose.model<IRestaurant>('Restaurant', RestaurantSchema);
