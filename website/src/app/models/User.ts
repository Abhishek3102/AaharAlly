import mongoose, { Document, Schema, Types } from "mongoose";

interface IUser extends Document {
  email: string;
  favoriteFoods: Types.ObjectId[];
  cart: { foodId: Types.ObjectId; quantity: number }[];
  age?: number;
  gender?: string;
  recommendedCategories?: string[];
  lastRecommendationDate?: Date;
  lastOrderDate?: Date;
  user_type?: string;
  cluster?: number;
  debugRecommendationData?: any;
}

const UserSchema = new Schema<IUser>({
  email: { type: String, unique: true, required: true },
  favoriteFoods: [{ type: Schema.Types.ObjectId, ref: "food" }],
  cart: [
    {
      foodId: { type: Schema.Types.ObjectId, ref: "food" },
      quantity: { type: Number, default: 1 },
    },
  ],
  age: { type: Number },
  gender: { type: String },
  recommendedCategories: [{ type: String }],
  lastRecommendationDate: { type: Date },
  lastOrderDate: { type: Date },
  user_type: { type: String },
  cluster: { type: Number },
  debugRecommendationData: { type: Schema.Types.Mixed },
}, { timestamps: true });

export const User =
  mongoose.models.User || mongoose.model<IUser>("User", UserSchema);
