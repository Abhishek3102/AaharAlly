import mongoose, { Document, Schema, Types } from "mongoose";

interface IUser extends Document {
  email: string;
  favoriteFoods: Types.ObjectId[];
  cart: { foodId: Types.ObjectId; quantity: number }[];
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
});

export const User =
  mongoose.models.User || mongoose.model<IUser>("User", UserSchema);
