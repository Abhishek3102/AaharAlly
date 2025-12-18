import mongoose, { Document, Schema } from 'mongoose';

export interface IAdmin extends Document {
    email: string;
    password: string;
    createdAt: Date;
}

const AdminSchema = new Schema<IAdmin>({
    email: {
        type: String,
        required: [true, 'Please provide an email'],
        unique: true,
    },
    password: {
        type: String,
        required: [true, 'Please provide a password'],
    },
    createdAt: {
        type: Date,
        default: Date.now,
    },
});

export const Admin = mongoose.models.Admin || mongoose.model<IAdmin>('Admin', AdminSchema);
