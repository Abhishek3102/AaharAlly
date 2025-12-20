from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from pymongo import MongoClient
import json, random, warnings
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# For LSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

app = Flask(__name__)

# MongoDB
client = MongoClient(os.getenv("MONGO_URL"))
db = client['aahar_ally_ml']
orders_col = db['orders']
reviews_col = db['reviews']
models_col = db['models']
cluster_pop_col = db['cluster_popularity']
rest_pop_col = db['restaurant_popularity']
users_col = db['users']

# Globals
imputer = None
scaler = None
gender_le = None
meal_ohe = None
pca = None
kmeans = None
tfidf = None
sent_clf = None
svd = None
user_index = None
item_index = None
item_reverse_index = None

# LSTM Globals
tokenizer = None
lstm_model = None
max_seq_len = 100


### Utilities
def to_jsonable(d):
    return json.loads(json.dumps(d, default=str))

def autolab_review_label(x):
    x = str(x)
    if any(word in x.lower() for word in ["not", "never", "no", "terrible", "bad", "awful", "worst", "disappointed"]):
        return 0
    return 1 if TextBlob(x).sentiment.polarity >= 0 else 0

def standardize_gender(g):
    g = str(g).strip().lower()
    if g in ["male", "m", "man", "boy"]: return "male"
    if g in ["female", "f", "woman", "girl"]: return "female"
    return "other"


### ---- Training Pipelines ----

def fit_preprocess_cluster(df):
    global imputer, scaler, gender_le, meal_ohe, pca, kmeans

    # Standardize gender
    df['gender'] = df['gender'].apply(standardize_gender)

    gender_le = LabelEncoder()
    gender_le.fit(["male", "female", "other"])  # fixed set
    df['gender_enc'] = gender_le.transform(df['gender'])

    meal_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    meal_encoded = meal_ohe.fit_transform(df[['meal_category']])
    meal_df = pd.DataFrame(meal_encoded, columns=meal_ohe.get_feature_names_out(['meal_category']))

    features = pd.concat([df[['age','gender_enc']].reset_index(drop=True),
                          meal_df.reset_index(drop=True)], axis=1)

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
        sub = df[df['cluster']==c]
        if sub.empty: continue
        counts = sub['meal_category'].value_counts()
        top = counts.head(10).index.tolist()
        cluster_pop_col.insert_one({'cluster': int(c), 'top_categories': top})
        res.append({'cluster': int(c), 'top_categories': top})
    return res


def fit_restaurant_popularity(df):
    rest_pop_col.delete_many({})
    out = []
    for rid, g in df.groupby('restaurant_id'):
        counts = g['meal_category'].value_counts().head(20)
        rest_pop_col.insert_one({
            'restaurant_id': str(rid),
            'top_categories': counts.index.tolist(),
            'counts': counts.tolist()
        })
        out.append({'restaurant_id': str(rid), 'top_categories': counts.index.tolist()})
    return out


def fit_sentiment(df):
    global tfidf, sent_clf
    texts = df['review'].fillna('').astype(str)
    y = texts.apply(autolab_review_label).astype(int)

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)

    sent_clf = LogisticRegression(max_iter=1000)
    sent_clf.fit(X, y)

    y_pred = sent_clf.predict(X)
    clf_report = classification_report(y, y_pred, output_dict=True)

    return {'trained_samples': int(X.shape[0]), 'accuracy_report': clf_report}


def predict_sentiment(texts):
    if tfidf is None or sent_clf is None:
        raise Exception("Sentiment model not trained yet")
    X = tfidf.transform(pd.Series(texts).astype(str))
    return sent_clf.predict_proba(X)[:,1]


### --------- LSTM Sentiment ------------

def fit_lstm_sentiment(df):
    global tokenizer, lstm_model, max_seq_len

    texts = df['review'].fillna('').astype(str).tolist()
    labels = [autolab_review_label(t) for t in texts]

    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_seq_len, padding='post', truncating='post')

    labels = np.array(labels)

    split_idx = int(0.8 * len(padded))
    X_train, X_test = padded[:split_idx], padded[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    lstm_model = Sequential([
        Embedding(input_dim=20000, output_dim=64, input_length=max_seq_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=2, batch_size=64,
                   validation_data=(X_test, y_test), verbose=1)

    loss, acc = lstm_model.evaluate(X_test, y_test, verbose=0)
    return {'trained_samples': len(padded), 'validation_accuracy': float(acc)}


def predict_lstm_sentiment(texts):
    if tokenizer is None or lstm_model is None:
        raise Exception("LSTM sentiment model not trained yet")
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_seq_len, padding='post', truncating='post')
    probs = lstm_model.predict(padded)
    return probs.flatten()


### ---- Collaborative Filtering ----
def fit_cf(df):
    global svd, user_index, item_index, item_reverse_index
    t = df.groupby(['user_id','meal_category']).size().reset_index(name='rating')
    if t.empty:
        return {'user_count': 0, 'item_count': 0, 'cf_metrics': {}}

    reader = Reader(rating_scale=(1, max(5, t['rating'].max())))
    data = Dataset.load_from_df(t[['user_id','meal_category','rating']], reader)
    trainset = data.build_full_trainset()

    svd = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(trainset)

    cf_eval = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)

    user_index = {uid:i for i,uid in enumerate(t['user_id'].unique())}
    items = t['meal_category'].unique().tolist()
    item_index = {it:i for i,it in enumerate(items)}
    item_reverse_index = {i:it for it,i in item_index.items()}

    return {
        'user_count': len(user_index),
        'item_count': len(item_index),
        'cf_metrics': {
            'RMSE': float(np.mean(cf_eval['test_rmse'])),
            'MAE': float(np.mean(cf_eval['test_mae']))
        }
    }


def rank_with_cf(user_id, candidates):
    if svd is None or not candidates:
        return candidates
    scores = []
    for it in candidates:
        try:
            est = svd.predict(str(user_id), str(it)).est
        except:
            est = 0.0
        scores.append((it, est))
    scores = [(c, s + random.uniform(-0.1, 0.1)) for c, s in scores]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores]


def build_sentiment_category_scores(df):
    if 'review' not in df.columns:
        return {}
    df2 = df[['meal_category','review']].dropna()
    if df2.empty: return {}
    probs = predict_sentiment(df2['review'])
    s = pd.DataFrame({'meal_category': df2['meal_category'].values, 'pos': probs})
    return s.groupby('meal_category')['pos'].mean().to_dict()


### ---- Helpers ----
def age_gender_to_cluster(age, gender, meal_category=None):
    if gender_le is None or meal_ohe is None or imputer is None or scaler is None or pca is None or kmeans is None:
        raise Exception("Clustering model not trained yet")
    g = standardize_gender(gender)
    g_enc = gender_le.transform([g])[0]
    row = pd.DataFrame({'age':[age], 'gender_enc':[g_enc], 'meal_category':[meal_category or "unknown"]})
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

# Persistence Hooks
import pickle
def save_models_to_db():
    print("Saving models to DB...")
    data = {
        'imputer': pickle.dumps(imputer) if imputer else None,
        'scaler': pickle.dumps(scaler) if scaler else None,
        'gender_le': pickle.dumps(gender_le) if gender_le else None,
        'meal_ohe': pickle.dumps(meal_ohe) if meal_ohe else None,
        'pca': pickle.dumps(pca) if pca else None,
        'kmeans': pickle.dumps(kmeans) if kmeans else None,
        'svd': pickle.dumps(svd) if svd else None,
        'cat_sentiment': (models_col.find_one({'model':'metadata'}) or {}).get('cat_sentiment')
    }
    models_col.delete_many({'model': 'core_components'})
    models_col.insert_one({'model': 'core_components', 'data': data})
    print("Models saved.")

def load_models_from_db():
    global imputer, scaler, gender_le, meal_ohe, pca, kmeans, svd
    doc = models_col.find_one({'model': 'core_components'})
    if doc:
        data = doc['data']
        imputer = pickle.loads(data['imputer']) if data.get('imputer') else None
        scaler = pickle.loads(data['scaler']) if data.get('scaler') else None
        gender_le = pickle.loads(data['gender_le']) if data.get('gender_le') else None
        meal_ohe = pickle.loads(data['meal_ohe']) if data.get('meal_ohe') else None
        pca = pickle.loads(data['pca']) if data.get('pca') else None
        kmeans = pickle.loads(data['kmeans']) if data.get('kmeans') else None
        svd = pickle.loads(data['svd']) if data.get('svd') else None
        print("Models loaded from DB.")

import threading

def run_training_pipeline(csv_path):
    with app.app_context():
        try:
            print("--- Starting Background Training ---")
            # Accept explicit path but default to train_data.csv
            p = csv_path
            
            # Try reading with header first
            df = pd.read_csv(p)
            if df.columns[0].lower() not in ["user_id", "uid"]:
                df = pd.read_csv(p, header=None,
                                names=["user_id","restaurant_id","age","gender","meal_category","review"])

            # Clean types
            df['gender'] = df['gender'].apply(standardize_gender)
            df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())

            df = fit_preprocess_cluster(df)
            fit_apriori_cluster_popularity(df)
            fit_restaurant_popularity(df)
            fit_sentiment(df)
            fit_cf(df)
            fit_lstm_sentiment(df)

            cat_sent = build_sentiment_category_scores(df)
            models_col.delete_many({'model':'metadata'})
            models_col.insert_one({'model': 'metadata', 'cat_sentiment': cat_sent})
            
            save_models_to_db()
            print("--- Background Training Completed Successfully ---")
        except Exception as e:
            print(f"--- Background Training Failed: {e} ---")

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        data = request.json or {}
        p = data.get('csv_path', 'train_data.csv')
        
        # Start background thread
        thread = threading.Thread(target=run_training_pipeline, args=(p,))
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Training started in background. Check logs for completion.'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



@app.route('/api/store_order', methods=['POST'])
def api_store_order():
    try:
        data = request.get_json()
        data['user_id'] = str(data.get('user_id'))
        data['restaurant_id'] = str(data.get('restaurant_id'))
        data['gender'] = standardize_gender(data.get('gender', "other"))
        orders_col.insert_one(data)
        users_col.update_one({'user_id': data['user_id']},
            {'$set':{'user_id':data['user_id'],'age':data.get('age'),'gender':data['gender']}}, upsert=True)
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


@app.route('/api/sentiment/predict_lstm', methods=['POST'])
def api_sentiment_predict_lstm():
    try:
        texts = request.json.get('texts', [])
        probs = predict_lstm_sentiment(texts)
        return jsonify({'success':True,'positive_probabilities_lstm': probs.tolist()})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        payload = request.get_json()
        user_id = str(payload.get('user_id'))
        age = float(payload.get('age'))
        gender = standardize_gender(payload.get('gender'))
        restaurant_id = str(payload.get('restaurant_id'))

        c = age_gender_to_cluster(age, gender)
        cluster_cats = pop_for_cluster(c)
        rest_cats = pop_for_restaurant(restaurant_id)
        hist = top_history_categories(user_id, n=10)

        cat_sent_map = (models_col.find_one({'model':'metadata'}) or {}).get('cat_sentiment', {})
        new_user = orders_col.count_documents({'user_id': user_id}) == 0

        if new_user:
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

# Load on startup
try:
    load_models_from_db()
except:
    pass

if __name__ == '__main__':
    # RENDER REQUIRES host='0.0.0.0'
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
