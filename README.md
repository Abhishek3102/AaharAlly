
# 🥗 Aahar Ally - Hybrid Intelligent Recommendation System

> **A Research-Driven Approach to Personalized Dining** — Combining Demographic Clustering, Collaborative Filtering, and Sentiment Analysis to bridge the gap between small restaurants and data-driven personalization.

---

## 🌟 The Core Innovation

Most food delivery platforms rely on generic popularity trends or simple discounts. **Aahar Ally** introduces a novel **Hybrid Recommendation Framework** optimized for the "Cold Start" problem and nuanced user preferences.

Based on the research paper *"AaharAlly: A Hybrid Restaurant Recommendation System Using Machine Learning and Sentiment Analysis"*, this platform dynamically adapts to users in three stages:

1.  **For New Users (The "Cold Start" Solution)**: Uses **Demographic Clustering (K-Means)**. We analyze age and gender to map users into existing taste clusters, offering relevant suggestions instantly without any order history.
2.  **For Returning Users**: Uses **Collaborative Filtering (SVD)** to refine recommendations based on individual interaction latent factors.
3.  **Sentiment-Driven Optimization**: We don't just count stars. We use **Bi-Directional LSTM** and Logistic Regression to analyze the *text* of reviews, allowing positive/negative sentiment to re-rank the final recommendations.

---

## 🚀 Key Features

### 🧠 1. The Hybrid AI Engine (Research Core)

*   **Demographic Clustering**: Users are segmented using **K-Means (k=6)** based on age and gender. This mirrors real-world dining patterns (e.g., college students vs. families have distinct aggregated preferences).
*   **Sentiment Analysis**: A dual-model approach using **TF-IDF + Logistic Regression** and **Deep Learning (LSTM)** to capture context in user reviews (e.g., distinguishing "spicy but good" from "spicy and inedible").
*   **Dynamic Retraining**: The system learns from every interaction, constantly refining the vectors used for clustering and recommendation.

<details>
<summary><strong>🩺 2. Dietary Intelligence (Safety Layer)</strong></summary>

*   **Health Conflict Filtering**: Beyond taste, Aahar Ally ensures safety. Users with conditions like **Diabetes** or **Hypertension** get a filtered menu.
*   **RAG Chatbot**: A context-aware assistant that can answer specific medical queries about the menu using accurate nutritional data.
</details>

<details>
<summary><strong>🎨 3. GenAI Plate Visualizer (Visual Confidence)</strong></summary>

*   **Real-Time Modification Previews**: Uses **Google Imagen 3** to visualize changes. If a user asks to "Remove cheese and add olives", the system generates a photorealistic preview of that specific modification to build trust.
</details>

<details>
<summary><strong>📊 4. Restaurant Analytics Dashboard</strong></summary>

*   **Cluster Insights**: Restaurants can see which demographic clusters are engaging most with their menu.
*   **Sentiment Trends**: Track how specific dishes are performing in terms of sentiment, not just sales volume.
</details>

---

## 🏗️ Technical Architecture

The project is structured as a Monorepo containing a High-Performance Frontend and a Research-Grade ML Backend.

### 🛠 Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Frontend** | Next.js 15, React 19, TailwindCSS, Framer Motion |
| **ML Engine** | Python, Flask, Scikit-Learn (KMeans, SVD), TensorFlow (LSTM) |
| **Database** | MongoDB Atlas (Shared Cluster) |
| **Vector DB** | Qdrant (for Semantic Search & RAG) |
| **Auth** | Clerk |

### ML Pipeline Flow
1.  **Ingestion**: Demographics & Order History → **Preprocessing** (KNN Imputer, OneHotEncoder).
2.  **Clustering**: PCA Dimension Reduction → KMeans Clustering.
3.  **Collaborative Filtering**: SVD Matrix Factorization on User-Dish interactions.
4.  **Ranking**: Recommendations are biased by Restaurant Popularity and re-ranked by Sentiment Scores.

---

## 📂 Project Structure

This repository is organized into three main workspaces.

### 1. `website/` (The Application Layer)
The Next.js application hosting the Client Interface and Admin Dashboard.

*   `src/app/`: App Router structure.
    *   `src/components/Recommendations.tsx`: The UI component that displays the personalized ML results.
    *   `src/app/models/HealthCache.js`: The bridge model connecting to the ML database.

### 2. `ML_Service/` (The Intelligence Brain)
A Python/Flask service deployed on **Render**. It performs the heavy lifting:

*   `app.py`: Exposes the `/api/train` endpoint.
*   **Functionality**:
    *   Fetches live data from MongoDB.
    *   Executes the **KMeans** and **SVD** pipelines.
    *   **Writes results** to the `aahar_ally_ml` database.

### 3. `ML/` (Research & Development)
Contains the original Jupyter notebooks used to validate the thesis:
*   `adding_accuracy_checks.py`: Scripts for validating the F1-score and Accuracy of the hybrid model (Achieved ~91.2% Accuracy in pilot studies).

---

## 🚀 Deployment

The project is deployed across two platforms:

1.  **Vercel** (`website` folder): Hosts the Frontend and API.
2.  **Render** (`ML_Service` folder): Hosts the Python ML Engine.

---

