
const axios = require('axios');

async function train() {
    try {
        console.log("Triggering ML training...");
        const res = await axios.post("https://aaharally.onrender.com/api/train", {
            csv_path: "train_data.csv"
        });
        console.log("Training response:", res.data);
    } catch (err) {
        console.error("Training failed:", err.message);
        if (err.response) {
            console.error("Status:", err.response.status);
            console.error("Data:", err.response.data);
        }
    }
}

train();
