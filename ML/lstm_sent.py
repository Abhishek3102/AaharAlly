from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset as SurpriseDataset, Reader, SVD
from pymongo import MongoClient
import json, random, warnings
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
app = Flask(__name__)

# MongoDB setup
client = MongoClient('mongodb+srv://betterpandey:z69UbLypqqusaeCK@aaharally.kz6zjsh.mongodb.net/?retryWrites=true&w=majority&appName=AaharAlly/')
db = client['aahar_ally_ml']
orders_col = db['orders']
reviews_col = db['reviews']
models_col = db['models']
cluster_pop_col = db['cluster_popularity']
rest_pop_col = db['restaurant_popularity']
users_col = db['users']

# Globals for preprocessing and models
imputer = scaler = gender_le = meal_ohe = pca = kmeans = None
tokenizer = lstm_model = None
svd = None
user_index = item_index = item_reverse_index = None

# LSTM & Tokenization settings
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_jsonable(d):
    return json.loads(json.dumps(d, default=str))

def autolab_review_label(x):
    p = (lambda t: torch.tensor([0]))(x)  # dummy override for clarity, using TextBlob above
    from textblob import TextBlob
    p = TextBlob(str(x)).sentiment.polarity
    return 1 if p >= 0 else 0

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return self.sigmoid(x)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return torch.LongTensor(self.texts[idx]), torch.FloatTensor([self.labels[idx]])

def train_model(model, loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

### ---- Training Pipelines ----

def fit_preprocess_cluster(df):
    global imputer, scaler, gender_le, meal_ohe, pca, kmeans
    gender_le = LabelEncoder()
    df['gender_enc'] = gender_le.fit_transform(df['gender'].astype(str))
    meal_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    meal_encoded = meal_ohe.fit_transform(df[['meal_category']])
    meal_df = pd.DataFrame(meal_encoded, columns=meal_ohe.get_feature_names_out(['meal_category']))
    features = pd.concat([df[['age', 'gender_enc']].reset_index(drop=True), meal_df], axis=1)
    imputer = KNNImputer(n_neighbors=3)
    features = imputer.fit_transform(features)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    pca = PCA(n_components=min(10, features.shape[1]), random_state=42)
    features = pca.fit_transform(features)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)
    return df

def fit_apriori_cluster_popularity(df):
    cluster_pop_col.delete_many({})
    res = []
    for c in sorted(df['cluster'].unique()):
        sub = df[df['cluster'] == c]
        if sub.empty:
            continue
        top = sub['meal_category'].value_counts().head(10).index.tolist()
        cluster_pop_col.insert_one({'cluster': int(c), 'top_categories': top})
        res.append({'cluster': int(c), 'top_categories': top})
    return res

def fit_restaurant_popularity(df):
    rest_pop_col.delete_many({})
    out = []
    for rid, g in df.groupby('restaurant_id'):
        counts = g['meal_category'].value_counts().head(20)
        rest_pop_col.insert_one({'restaurant_id': str(rid),
                                 'top_categories': counts.index.tolist(),
                                 'counts': counts.tolist()})
        out.append({'restaurant_id': str(rid), 'top_categories': counts.index.tolist()})
    return out

def fit_sentiment(df):
    global tokenizer, lstm_model
    texts = df['review'].fillna('').astype(str)
    y = texts.apply(autolab_review_label).astype(int).values
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    X_train, X_val, y_train, _ = train_test_split(padded, y, test_size=0.1, random_state=42)
    dataset = TextDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    vocab_size = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)
    lstm_model = LSTMSentimentClassifier(vocab_size, EMBEDDING_DIM, 128, 1).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    train_model(lstm_model, loader, criterion, optimizer)
    return {'trained_samples': len(dataset)}

def predict_sentiment(texts):
    global tokenizer, lstm_model
    lstm_model.eval()
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    inputs = torch.LongTensor(padded).to(device)
    with torch.no_grad():
        outputs = lstm_model(inputs).cpu().numpy().flatten()
    return outputs

def fit_cf(df):
    global svd, user_index, item_index, item_reverse_index
    t = df.groupby(['user_id', 'meal_category']).size().reset_index(name='rating')
    reader = Reader(rating_scale=(1, max(5, t['rating'].max())))
    data = SurpriseDataset.load_from_df(t[['user_id', 'meal_category', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(trainset)
    user_index = {uid: i for i, uid in enumerate(t['user_id'].unique())}
    items = t['meal_category'].unique().tolist()
    item_index = {it: i for i, it in enumerate(items)}
    item_reverse_index = {i: it for it, i in item_index.items()}
    return {'user_count': len(user_index), 'item_count': len(item_index)}

def rank_with_cf(user_id, candidates):
    if svd is None or not candidates:
        return candidates
    scores = []
    for it in candidates:
        try:
            est = svd.predict(str(user_id), str(it)).est
        except:
            est = 0.0
        scores.append((it, est + random.uniform(-0.1, 0.1)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scores]

def build_sentiment_category_scores(df):
    if 'review' not in df.columns:
        return {}
    df2 = df[['meal_category', 'review']].dropna()
    if df2.empty:
        return {}
    probs = predict_sentiment(df2['review'].tolist())
    s = pd.DataFrame({'meal_category': df2['meal_category'].values, 'pos': probs})
    return s.groupby('meal_category')['pos'].mean().to_dict()


def age_gender_to_cluster(age, gender, meal_category=None):
    g = gender_le.transform([gender])[0]
    row = pd.DataFrame({'age':[age], 'gender_enc':[g], 'meal_category':[meal_category or "unknown"]})
    meal_vec = meal_ohe.transform(row[['meal_category']])
    feat = np.concatenate([row[['age','gender_enc']].values, meal_vec], axis=1)
    feat = imputer.transform(feat)
    feat = scaler.transform(feat)
    feat = pca.transform(feat)
    return int(kmeans.predict(feat)[0])

def pop_for_cluster(c):
    r = cluster_pop_col.find_one({'cluster': int(c)})
    return r['top_categories'] if r and 'top_categories' in r else []

def pop_for_restaurant(rid):
    r = rest_pop_col.find_one({'restaurant_id': str(rid)})
    return r['top_categories'] if r and 'top_categories' in r else []

def top_history_categories(uid, n=10):
    cur = orders_col.aggregate([
        {'$match': {'user_id': str(uid)}},
        {'$group': {'_id': '$meal_category', 'cnt': {'$sum': 1}}},
        {'$sort': {'cnt': -1}}, {'$limit': n}
    ])
    return [x['_id'] for x in cur]

def apply_sentiment_rerank(candidates, cat_sent_map):
    if not candidates or not cat_sent_map: return candidates
    scored = [(c, cat_sent_map.get(c, 0.5) + random.uniform(-0.05,0.05)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored]

### ---- API ----
@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        p = request.json.get('csv_path', 'train_data.csv')
        df = pd.read_csv(p)
        if 'review' not in df.columns: df['review'] = ''
        df = fit_preprocess_cluster(df)
        a = fit_apriori_cluster_popularity(df)
        b = fit_restaurant_popularity(df)
        c = fit_sentiment(df)
        d = fit_cf(df)
        cat_sent = build_sentiment_category_scores(df)
        models_col.delete_many({})
        models_col.insert_one({'model':'metadata','cat_sentiment': cat_sent})
        return jsonify({'success':True,'clusters':a,'restaurants':b,'sentiment':c,'cf':d})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/store_order', methods=['POST'])
def api_store_order():
    try:
        data = request.get_json()
        data['user_id'] = str(data.get('user_id'))
        data['restaurant_id'] = str(data.get('restaurant_id'))
        orders_col.insert_one(data)
        users_col.update_one({'user_id': data['user_id']},
            {'$set':{'user_id':data['user_id'],'age':data.get('age'),'gender':data.get('gender')}}, upsert=True)
        return jsonify({'success':True})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/sentiment/predict', methods=['POST'])
def api_sentiment_predict():
    try:
        texts = request.json.get('texts', [])
        probs = predict_sentiment(texts)
        return jsonify({'success':True,'positive_probabilities': probs.tolist()})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        payload = request.get_json()
        user_id = str(payload.get('user_id'))
        age, gender, restaurant_id = float(payload.get('age')), str(payload.get('gender')), str(payload.get('restaurant_id'))

        c = age_gender_to_cluster(age, gender)
        cluster_cats, rest_cats, hist = pop_for_cluster(c), pop_for_restaurant(restaurant_id), top_history_categories(user_id, n=10)
        cat_sent_map = (models_col.find_one({'model':'metadata'}) or {}).get('cat_sentiment', {})
        new_user = orders_col.count_documents({'user_id': user_id}) == 0

        if new_user:
            # weighted blend with randomness
            cand = list(set(cluster_cats) | set(rest_cats))
            random.shuffle(cand)
            cand = apply_sentiment_rerank(cand, cat_sent_map)
            cand = rank_with_cf(user_id, cand)
            return jsonify({'success':True,'user_type':'new','cluster':c,'recommendations':cand[:10]})
        else:
            base = list(dict.fromkeys(hist + cluster_cats + rest_cats))
            if hist: base = [x for x in base if x in set(hist + cluster_cats)]
            random.shuffle(base)
            base = apply_sentiment_rerank(base, cat_sent_map)
            base = rank_with_cf(user_id, base)
            return jsonify({'success':True,'user_type':'returning','cluster':c,'recommendations':base[:10]})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

if __name__ == '__main__':
    app.run(debug=False)
