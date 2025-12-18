const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
require('dotenv').config({ path: '.env.local' }); // Try .env.local first
require('dotenv').config(); // Fallback to .env

// --- CONFIGURATION ---
// REPLACE THESE VALUES WITH YOUR DESIRED CREDENTIALS
const ADMIN_EMAIL = process.env.ADMIN_EMAIL;
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD;

if (!ADMIN_EMAIL || !ADMIN_PASSWORD) {
  console.error("Error: ADMIN_EMAIL or ADMIN_PASSWORD is not defined in environment variables.");
  process.exit(1);
}
// ---------------------

// MongoDB Connection String
const MONGO_URL = process.env.MONGO_URL;

if (!MONGO_URL) {
  console.error("Error: MONGO_URL is not defined in environment variables.");
  process.exit(1);
}

const adminSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
});

const Admin = mongoose.models.Admin || mongoose.model('Admin', adminSchema);

async function createAdmin() {
  try {
    console.log("Connecting to MongoDB...");
    await mongoose.connect(MONGO_URL);
    console.log("Connected.");

    const existingAdmin = await Admin.findOne({ email: ADMIN_EMAIL });
    if (existingAdmin) {
      console.log(`Admin with email ${ADMIN_EMAIL} already exists.`);
      process.exit(0);
    }

    console.log("Hashing password...");
    const hashedPassword = await bcrypt.hash(ADMIN_PASSWORD, 10);

    console.log("Creating admin...");
    await Admin.create({
      email: ADMIN_EMAIL,
      password: hashedPassword,
    });

    console.log("Admin created successfully!");
    console.log(`Email: ${ADMIN_EMAIL}`);
    console.log(`Password: ${ADMIN_PASSWORD}`);
    
    process.exit(0);
  } catch (error) {
    console.error("Error creating admin:", error);
    process.exit(1);
  }
}

createAdmin();
