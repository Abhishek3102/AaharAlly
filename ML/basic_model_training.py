
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import os

# Set plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Load Data
# Using the path from generate_training_data.py
csv_path = "c:/aaharally/AaharAlly/ML_Service/train_data.csv"
if not os.path.exists(csv_path):
    # Fallback to local ML folder if service folder is missing
    csv_path = "train_data.csv"

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Data loaded: {len(df)} rows.")

# ---------------------------------------------------------
# 2. K-Means Demographics Clustering
# ---------------------------------------------------------
print("\n--- Starting K-Means Clustering ---")
# Preprocess
cluster_df = df[['age', 'gender']].drop_duplicates().copy()
le = LabelEncoder()
cluster_df['gender_enc'] = le.fit_transform(cluster_df['gender'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df[['age', 'gender_enc']])

# Train
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
cluster_df['cluster'] = kmeans.fit_predict(X_scaled)

# Metrics
sil_avg = silhouette_score(X_scaled, cluster_df['cluster'])
print(f"K-Means Silhouette Score: {sil_avg:.4f}")

# Visualization (PCA to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
cluster_df['pca1'] = X_pca[:, 0]
cluster_df['pca2'] = X_pca[:, 1]

plt.figure()
sns.scatterplot(data=cluster_df, x='pca1', y='pca2', hue='cluster', palette='viridis', s=100)
plt.title("K-Means Demographics Clustering (PCA Projection)")
plt.savefig("kmeans_clusters.png")
print("Saved kmeans_clusters.png")

# ---------------------------------------------------------
# 3. SVD Collaborative Filtering
# ---------------------------------------------------------
print("\n--- Starting SVD Collaborative Filtering ---")
# Prepare data for Surprise
# We use interaction counts as 'rating'
inter_df = df.groupby(['user_id', 'meal_category']).size().reset_index(name='count')
reader = Reader(rating_scale=(1, inter_df['count'].max()))
data = Dataset.load_from_df(inter_df[['user_id', 'meal_category', 'count']], reader)

# Cross-validate
svd = SVD()
results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Metrics Plot
metrics = ['RMSE', 'MAE']
scores = [np.mean(results['test_rmse']), np.mean(results['test_mae'])]

plt.figure()
sns.barplot(x=metrics, y=scores, palette='magma')
plt.title("SVD Collaborative Filtering Evaluation")
plt.ylabel("Score")
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
plt.savefig("svd_metrics.png")
print("Saved svd_metrics.png")

# ---------------------------------------------------------
# 4. Bi-LSTM Sentiment Analysis
# ---------------------------------------------------------
print("\n--- Starting Bi-LSTM Sentiment Analysis ---")

# Heuristic Labeling (Simulating Sentiment)
def get_label(text):
    pos_words = ["loved", "amazing", "best", "recommend", "delicious", "perfect", "delightful"]
    if any(word in text.lower() for word in pos_words):
        return 1
    return 0

df['label'] = df['review'].apply(get_label)

# Preprocessing
max_words = 5000
max_len = 50
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
y = df['label'].values

# Split
split = int(0.8 * len(y))
X_train, X_test = X_padded[:split], X_padded[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = Sequential([
    Embedding(max_words, 64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training Bi-LSTM...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig("lstm_history.png")
print("Saved lstm_history.png")

# Final Metrics
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nModel training complete! All plots saved in the current directory.")
