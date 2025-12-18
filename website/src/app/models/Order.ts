import mongoose, { Document, Schema, Types } from 'mongoose';

export interface IOrder extends Document {
    userId: Types.ObjectId;
    items: {
        foodId: Types.ObjectId;
        quantity: number;
        price: number;
    }[];
    totalAmount: number;
    status: 'pending' | 'completed' | 'cancelled';
    createdAt: Date;
    updatedAt: Date;
}

const OrderSchema = new Schema<IOrder>(
    {
        userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
        items: [
            {
                foodId: { type: Schema.Types.ObjectId, ref: 'food', required: true },
                quantity: { type: Number, required: true },
                price: { type: Number, required: true },
            },
        ],
        totalAmount: { type: Number, required: true },
        status: {
            type: String,
            enum: ['pending', 'completed', 'cancelled'],
            default: 'pending',
        },
    },
    { timestamps: true }
);

export const Order = mongoose.models.Order || mongoose.model<IOrder>('Order', OrderSchema);
