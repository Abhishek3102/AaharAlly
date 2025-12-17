# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from textblob import TextBlob
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from mlxtend.frequent_patterns import apriori
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from pymongo import MongoClient
# from bson import ObjectId
# import json
# import warnings
# warnings.filterwarnings("ignore")

# app = Flask(__name__)

# # client = MongoClient('mongodb+srv://maurya48ashish:Ashish48Maurya@cluster0.w5ltbks.mongodb.net/')///////
# client = MongoClient('mongodb+srv://betterpandey:z69UbLypqqusaeCK@aaharally.kz6zjsh.mongodb.net/?retryWrites=true&w=majority&appName=AaharAlly/')
# db = client['aahar_ally_ml']
# orders_col = db['orders']
# reviews_col = db['reviews']
# models_col = db['models']
# cluster_pop_col = db['cluster_popularity']
# rest_pop_col = db['restaurant_popularity']
# users_col = db['users']

# imputer = None
# scaler = None
# gender_le = None
# kmeans = None
# cluster_map = None
# tfidf = None
# sent_clf = None
# svd = None
# user_index = None
# item_index = None
# item_reverse_index = None

# def to_jsonable(d):
#     return json.loads(json.dumps(d, default=str))

# def autolab_review_label(x):
#     p = TextBlob(str(x)).sentiment.polarity
#     if p > 0: return 1
#     if p < 0: return 0
#     return 1

# def fit_preprocess_cluster(df):
#     global imputer, scaler, gender_le, kmeans
#     x = df[['age','gender']].copy()
#     gender_le = LabelEncoder()
#     x['gender'] = gender_le.fit_transform(x['gender'].astype(str))
#     imputer = KNNImputer(n_neighbors=3)
#     x[['age','gender']] = imputer.fit_transform(x[['age','gender']])
#     scaler = MinMaxScaler()
#     x[['age']] = scaler.fit_transform(x[['age']])
#     kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
#     df['cluster'] = kmeans.fit_predict(x[['age','gender']])
#     return df

# def fit_apriori_cluster_popularity(df):
#     cluster_pop_col.delete_many({})
#     res = []
#     for c in sorted(df['cluster'].unique()):
#         sub = df[df['cluster']==c]
#         if sub.empty: 
#             continue
#         basket = pd.crosstab(sub['user_id'], sub['meal_category']).clip(upper=1)
#         if basket.shape[0]==0 or basket.shape[1]==0:
#             continue
#         freq = apriori(basket, min_support=0.05, use_colnames=True)
#         singles = freq[freq['itemsets'].apply(lambda s: len(s)==1)].copy()
#         if singles.empty:
#             counts = basket.sum().sort_values(ascending=False)
#             top = counts.head(10).index.tolist()
#         else:
#             singles['item'] = singles['itemsets'].apply(lambda s: list(s)[0])
#             top = singles.sort_values('support', ascending=False)['item'].head(10).tolist()
#         cluster_pop_col.insert_one({'cluster': int(c), 'top_categories': top})
#         res.append({'cluster': int(c), 'top_categories': top})
#     return res

# def fit_restaurant_popularity(df):
#     rest_pop_col.delete_many({})
#     out = []
#     for rid, g in df.groupby('restaurant_id'):
#         counts = g['meal_category'].value_counts().head(20)
#         rest_pop_col.insert_one({'restaurant_id': str(rid), 'top_categories': counts.index.tolist(), 'counts': counts.tolist()})
#         out.append({'restaurant_id': str(rid), 'top_categories': counts.index.tolist()})
#     return out

# def fit_sentiment(df):
#     global tfidf, sent_clf
#     texts = df['review'].fillna('').astype(str)
#     y = texts.apply(autolab_review_label).astype(int)
#     tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
#     X = tfidf.fit_transform(texts)
#     sent_clf = LogisticRegression(max_iter=1000)
#     sent_clf.fit(X, y)
#     return {'trained_samples': int(X.shape[0])}

# def predict_sentiment(texts):
#     X = tfidf.transform(pd.Series(texts).astype(str))
#     proba = sent_clf.predict_proba(X)[:,1]
#     return proba

# def fit_cf(df):
#     global svd, user_index, item_index, item_reverse_index
#     t = df.groupby(['user_id','meal_category']).size().reset_index(name='rating')
#     reader = Reader(rating_scale=(1, t['rating'].max() if t['rating'].max()>1 else 5))
#     data = Dataset.load_from_df(t[['user_id','meal_category','rating']], reader)
#     trainset = data.build_full_trainset()
#     svd = SVD(n_factors=50, n_epochs=20, random_state=42)
#     svd.fit(trainset)
#     user_index = {uid:i for i,uid in enumerate(t['user_id'].unique())}
#     items = t['meal_category'].unique().tolist()
#     item_index = {it:i for i,it in enumerate(items)}
#     item_reverse_index = {i:it for it,i in item_index.items()}
#     return {'user_count': len(user_index), 'item_count': len(item_index)}

# def rank_with_cf(user_id, candidates):
#     if svd is None or len(candidates)==0:
#         return candidates
#     scores = []
#     for it in candidates:
#         try:
#             est = svd.predict(str(user_id), str(it)).est
#         except:
#             est = 0.0
#         scores.append((it, est))
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return [s[0] for s in scores]

# def build_sentiment_category_scores(df):
#     if 'review' not in df.columns:
#         return {}
#     df2 = df[['meal_category','review']].dropna()
#     if df2.empty:
#         return {}
#     probs = predict_sentiment(df2['review'])
#     s = pd.DataFrame({'meal_category': df2['meal_category'].values, 'pos': probs})
#     cat = s.groupby('meal_category')['pos'].mean()
#     return cat.to_dict()

# def load_csv(path):
#     df = pd.read_csv(path)
#     req = ['user_id','restaurant_id','age','gender','meal_category']
#     for c in req:
#         if c not in df.columns:
#             raise ValueError(f"missing column: {c}")
#     if 'review' not in df.columns:
#         df['review'] = ''
#     return df

# def age_gender_to_cluster(age, gender):
#     x = pd.DataFrame({'age':[age], 'gender':[gender]})
#     g = gender_le.transform(x['gender'].astype(str))
#     a = imputer.transform(np.array([[age, g[0]]]))[0][0]
#     a2 = scaler.transform(np.array(a).reshape(-1,1))[0][0]
#     pred = kmeans.predict(np.array([[a2, g[0]]]))
#     return int(pred[0])

# def pop_for_cluster(c):
#     r = cluster_pop_col.find_one({'cluster': int(c)})
#     if r and 'top_categories' in r:
#         return r['top_categories']
#     return []

# def pop_for_restaurant(rid):
#     r = rest_pop_col.find_one({'restaurant_id': str(rid)})
#     if r and 'top_categories' in r:
#         return r['top_categories']
#     return []

# def top_history_categories(uid, n=10):
#     cur = orders_col.aggregate([
#         {'$match': {'user_id': str(uid)}},
#         {'$group': {'_id': '$meal_category', 'cnt': {'$sum': 1}}},
#         {'$sort': {'cnt': -1}},
#         {'$limit': n}
#     ])
#     return [x['_id'] for x in cur]

# def apply_sentiment_rerank(candidates, cat_sent_map):
#     if not candidates or not cat_sent_map:
#         return candidates
#     scored = [(c, cat_sent_map.get(c, 0.5)) for c in candidates]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [s[0] for s in scored]

# @app.route('/api/train', methods=['POST'])
# def api_train():
#     try:
#         p = request.json.get('csv_path', 'train_data.csv')
#         df = load_csv(p)
#         df = fit_preprocess_cluster(df)
#         a = fit_apriori_cluster_popularity(df)
#         b = fit_restaurant_popularity(df)
#         c = fit_sentiment(df)
#         d = fit_cf(df)
#         cat_sent = build_sentiment_category_scores(df)
#         models_col.delete_many({})
#         models_col.insert_one({'model':'metadata','cat_sentiment': cat_sent})
#         return jsonify({'success':True,'clusters':a,'restaurants':b,'sentiment':c,'cf':d})
#     except Exception as e:
#         return jsonify({'success':False,'error':str(e)})

# @app.route('/api/store_order', methods=['POST'])
# def api_store_order():
#     try:
#         data = request.get_json()
#         data['user_id'] = str(data.get('user_id'))
#         data['restaurant_id'] = str(data.get('restaurant_id'))
#         orders_col.insert_one(data)
#         users_col.update_one({'user_id': data['user_id']},{'$set':{'user_id':data['user_id'],'age':data.get('age'),'gender':data.get('gender')}},upsert=True)
#         return jsonify({'success':True})
#     except Exception as e:
#         return jsonify({'success':False,'error':str(e)})

# @app.route('/api/sentiment/predict', methods=['POST'])
# def api_sentiment_predict():
#     try:
#         texts = request.json.get('texts', [])
#         probs = predict_sentiment(texts)
#         return jsonify({'success':True,'positive_probabilities': probs.tolist()})
#     except Exception as e:
#         return jsonify({'success':False,'error':str(e)})

# @app.route('/api/recommend', methods=['POST'])
# def api_recommend():
#     try:
#         payload = request.get_json()
#         user_id = str(payload.get('user_id'))
#         age = float(payload.get('age'))
#         gender = str(payload.get('gender'))
#         restaurant_id = str(payload.get('restaurant_id'))
#         c = age_gender_to_cluster(age, gender)
#         cluster_cats = pop_for_cluster(c)
#         rest_cats = pop_for_restaurant(restaurant_id)
#         hist = top_history_categories(user_id, n=10)
#         cat_sent_doc = models_col.find_one({'model':'metadata'}) or {}
#         cat_sent_map = cat_sent_doc.get('cat_sentiment', {})
#         new_user = orders_col.count_documents({'user_id': user_id}) == 0
#         if new_user:
#             cand = [x for x in cluster_cats if x in rest_cats] or cluster_cats[:10] or rest_cats[:10]
#             cand = apply_sentiment_rerank(cand, cat_sent_map)
#             cand = rank_with_cf(user_id, cand)
#             return jsonify({'success':True,'user_type':'new','cluster':c,'recommendations':cand[:10]})
#         else:
#             base = list(dict.fromkeys(hist + cluster_cats + rest_cats))
#             if hist:
#                 base = [x for x in base if x in set(hist + cluster_cats)]
#             base = apply_sentiment_rerank(base, cat_sent_map)
#             base = rank_with_cf(user_id, base)
#             return jsonify({'success':True,'user_type':'returning','cluster':c,'recommendations':base[:10]})
#     except Exception as e:
#         return jsonify({'success':False,'error':str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from textblob import TextBlob
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from mlxtend.frequent_patterns import apriori
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from pymongo import MongoClient
# from bson import ObjectId
# import json
# import warnings
# warnings.filterwarnings("ignore")

# app = Flask(__name__)

# client = MongoClient('mongodb+srv://betterpandey:z69UbLypqqusaeCK@aaharally.kz6zjsh.mongodb.net/?retryWrites=true&w=majority&appName=AaharAlly/')
# db = client['aahar_ally_ml']
# orders_col = db['orders']
# reviews_col = db['reviews']
# models_col = db['models']
# cluster_pop_col = db['cluster_popularity']
# rest_pop_col = db['restaurant_popularity']
# users_col = db['users']

# imputer = None
# scaler = None
# gender_le = None
# meal_ohe = None   ### UPDATED
# pca = None        ### UPDATED
# kmeans = None
# cluster_map = None
# tfidf = None
# sent_clf = None
# svd = None
# user_index = None
# item_index = None
# item_reverse_index = None

# def to_jsonable(d):
#     return json.loads(json.dumps(d, default=str))

# def autolab_review_label(x):
#     p = TextBlob(str(x)).sentiment.polarity
#     if p > 0: return 1
#     if p < 0: return 0
#     return 1

# ### UPDATED
# def fit_preprocess_cluster(df):
#     """
#     Improved clustering:
#     Uses age + gender + meal categories (one-hot) so that clusters
#     reflect both demographics and eating patterns.
#     """
#     global imputer, scaler, gender_le, meal_ohe, pca, kmeans

#     # Encode gender
#     gender_le = LabelEncoder()
#     df['gender_enc'] = gender_le.fit_transform(df['gender'].astype(str))

#     # One-hot encode meal categories
#     meal_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
#     meal_encoded = meal_ohe.fit_transform(df[['meal_category']])
#     meal_df = pd.DataFrame(meal_encoded, columns=meal_ohe.get_feature_names_out(['meal_category']))

#     # Combine features
#     features = pd.concat([df[['age','gender_enc']].reset_index(drop=True), meal_df.reset_index(drop=True)], axis=1)

#     # Handle missing values
#     imputer = KNNImputer(n_neighbors=3)
#     features = imputer.fit_transform(features)

#     # Scale
#     scaler = MinMaxScaler()
#     features = scaler.fit_transform(features)

#     # Dimensionality reduction (optional but helps)
#     pca = PCA(n_components=min(10, features.shape[1]), random_state=42)
#     features = pca.fit_transform(features)

#     # KMeans
#     kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
#     df['cluster'] = kmeans.fit_predict(features)

#     return df

# ### UPDATED
# def fit_apriori_cluster_popularity(df):
#     """
#     For each cluster, get top meal categories based on actual frequency.
#     This ensures clusters have distinct top categories.
#     """
#     cluster_pop_col.delete_many({})
#     res = []
#     for c in sorted(df['cluster'].unique()):
#         sub = df[df['cluster']==c]
#         if sub.empty:
#             continue

#         # Frequency counts within this cluster
#         counts = sub['meal_category'].value_counts()
#         top = counts.head(10).index.tolist()

#         cluster_pop_col.insert_one({'cluster': int(c), 'top_categories': top})
#         res.append({'cluster': int(c), 'top_categories': top})

#     return res

# def fit_restaurant_popularity(df):
#     rest_pop_col.delete_many({})
#     out = []
#     for rid, g in df.groupby('restaurant_id'):
#         counts = g['meal_category'].value_counts().head(20)
#         rest_pop_col.insert_one({'restaurant_id': str(rid), 'top_categories': counts.index.tolist(), 'counts': counts.tolist()})
#         out.append({'restaurant_id': str(rid), 'top_categories': counts.index.tolist()})
#     return out

# def fit_sentiment(df):
#     global tfidf, sent_clf
#     texts = df['review'].fillna('').astype(str)
#     y = texts.apply(autolab_review_label).astype(int)
#     tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
#     X = tfidf.fit_transform(texts)
#     sent_clf = LogisticRegression(max_iter=1000)
#     sent_clf.fit(X, y)
#     return {'trained_samples': int(X.shape[0])}

# def predict_sentiment(texts):
#     X = tfidf.transform(pd.Series(texts).astype(str))
#     proba = sent_clf.predict_proba(X)[:,1]
#     return proba

# def fit_cf(df):
#     global svd, user_index, item_index, item_reverse_index
#     t = df.groupby(['user_id','meal_category']).size().reset_index(name='rating')
#     reader = Reader(rating_scale=(1, t['rating'].max() if t['rating'].max()>1 else 5))
#     data = Dataset.load_from_df(t[['user_id','meal_category','rating']], reader)
#     trainset = data.build_full_trainset()
#     svd = SVD(n_factors=50, n_epochs=20, random_state=42)
#     svd.fit(trainset)
#     user_index = {uid:i for i,uid in enumerate(t['user_id'].unique())}
#     items = t['meal_category'].unique().tolist()
#     item_index = {it:i for i,it in enumerate(items)}
#     item_reverse_index = {i:it for it,i in item_index.items()}
#     return {'user_count': len(user_index), 'item_count': len(item_index)}

# def rank_with_cf(user_id, candidates):
#     if svd is None or len(candidates)==0:
#         return candidates
#     scores = []
#     for it in candidates:
#         try:
#             est = svd.predict(str(user_id), str(it)).est
#         except:
#             est = 0.0
#         scores.append((it, est))
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return [s[0] for s in scores]

# def build_sentiment_category_scores(df):
#     if 'review' not in df.columns:
#         return {}
#     df2 = df[['meal_category','review']].dropna()
#     if df2.empty:
#         return {}
#     probs = predict_sentiment(df2['review'])
#     s = pd.DataFrame({'meal_category': df2['meal_category'].values, 'pos': probs})
#     cat = s.groupby('meal_category')['pos'].mean()
#     return cat.to_dict()

# def load_csv(path):
#     df = pd.read_csv(path)
#     req = ['user_id','restaurant_id','age','gender','meal_category']
#     for c in req:
#         if c not in df.columns:
#             raise ValueError(f"missing column: {c}")
#     if 'review' not in df.columns:
#         df['review'] = ''
#     return df

# ### UPDATED
# def age_gender_to_cluster(age, gender, meal_category=None):
#     """
#     Predicts cluster for a given user using demographics + optional meal_category.
#     """
#     g = gender_le.transform([gender])[0]
#     row = pd.DataFrame({'age':[age], 'gender_enc':[g], 'meal_category':[meal_category or "unknown"]})

#     # Encode meal_category
#     meal_vec = meal_ohe.transform(row[['meal_category']])
#     feat = np.concatenate([row[['age','gender_enc']].values, meal_vec], axis=1)

#     feat = imputer.transform(feat)
#     feat = scaler.transform(feat)
#     feat = pca.transform(feat)

#     pred = kmeans.predict(feat)
#     return int(pred[0])

# def pop_for_cluster(c):
#     r = cluster_pop_col.find_one({'cluster': int(c)})
#     if r and 'top_categories' in r:
#         return r['top_categories']
#     return []

# def pop_for_restaurant(rid):
#     r = rest_pop_col.find_one({'restaurant_id': str(rid)})
#     if r and 'top_categories' in r:
#         return r['top_categories']
#     return []

# def top_history_categories(uid, n=10):
#     cur = orders_col.aggregate([
#         {'$match': {'user_id': str(uid)}},
#         {'$group': {'_id': '$meal_category', 'cnt': {'$sum': 1}}},
#         {'$sort': {'cnt': -1}},
#         {'$limit': n}
#     ])
#     return [x['_id'] for x in cur]

# def apply_sentiment_rerank(candidates, cat_sent_map):
#     if not candidates or not cat_sent_map:
#         return candidates
#     scored = [(c, cat_sent_map.get(c, 0.5)) for c in candidates]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [s[0] for s in scored]

# @app.route('/api/train', methods=['POST'])
# def api_train():
#     try:
#         p = request.json.get('csv_path', 'train_data.csv')
#         df = load_csv(p)
#         df = fit_preprocess_cluster(df)
#         a = fit_apriori_cluster_popularity(df)
#         b = fit_restaurant_popularity(df)
#         c = fit_sentiment(df)
#         d = fit_cf(df)
#         cat_sent = build_sentiment_category_scores(df)
#         models_col.delete_many({})
#         models_col.insert_one({'model':'metadata','cat_sentiment': cat_sent})
#         return jsonify({'success':True,'clusters':a,'restaurants':b,'sentiment':c,'cf':d})
#     except Exception as e:
#         return jsonify({'success':False,'error':str(e)})

# @app.route('/api/store_order', methods=['POST'])
# def api_store_order():
#     try:
#         data = request.get_json()
#         data['user_id'] = str(data.get('user_id'))
#         data['restaurant_id'] = str(data.get('restaurant_id'))
#         orders_col.insert_one(data)
#         users_col.update_one({'user_id': data['user_id']},{'$set':{'user_id':data['user_id'],'age':data.get('age'),'gender':data.get('gender')}},upsert=True)
#         return jsonify({'success':True})
#     except Exception as e:
#         return jsonify({'success':False,'error':str(e)})

# @app.route('/api/sentiment/predict', methods=['POST'])
# def api_sentiment_predict():
#     try:
#         texts = request.json.get('texts', [])
#         probs = predict_sentiment(texts)
#         return jsonify({'success':True,'positive_probabilities': probs.tolist()})
#     except Exception as e:
#         return jsonify({'success':False,'error':str(e)})

# @app.route('/api/recommend', methods=['POST'])
# def api_recommend():
#     try:
#         payload = request.get_json()
#         user_id = str(payload.get('user_id'))
#         age = float(payload.get('age'))
#         gender = str(payload.get('gender'))
#         restaurant_id = str(payload.get('restaurant_id'))
#         c = age_gender_to_cluster(age, gender)
#         cluster_cats = pop_for_cluster(c)
#         rest_cats = pop_for_restaurant(restaurant_id)
#         hist = top_history_categories(user_id, n=10)
#         cat_sent_doc = models_col.find_one({'model':'metadata'}) or {}
#         cat_sent_map = cat_sent_doc.get('cat_sentiment', {})
#         new_user = orders_col.count_documents({'user_id': user_id}) == 0
#         if new_user:
#             cand = [x for x in cluster_cats if x in rest_cats] or cluster_cats[:10] or rest_cats[:10]
#             cand = apply_sentiment_rerank(cand, cat_sent_map)
#             cand = rank_with_cf(user_id, cand)
#             return jsonify({'success':True,'user_type':'new','cluster':c,'recommendations':cand[:10]})
#         else:
#             base = list(dict.fromkeys(hist + cluster_cats + rest_cats))
#             if hist:
#                 base = [x for x in base if x in set(hist + cluster_cats)]
#             base = apply_sentiment_rerank(base, cat_sent_map)
#             base = rank_with_cf(user_id, base)
#             return jsonify({'success':True,'user_type':'returning','cluster':c,'recommendations':base[:10]})
#     except Exception as e:
#         return jsonify({'success':False,'error':str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)




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
from surprise import Dataset, Reader, SVD
from pymongo import MongoClient
import json, random, warnings
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

app = Flask(__name__)

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

### Utilities
def to_jsonable(d):
    return json.loads(json.dumps(d, default=str))

def autolab_review_label(x):
    p = TextBlob(str(x)).sentiment.polarity
    return 1 if p >= 0 else 0

### ---- Training Pipelines ----
def fit_preprocess_cluster(df):
    global imputer, scaler, gender_le, meal_ohe, pca, kmeans
    gender_le = LabelEncoder()
    df['gender_enc'] = gender_le.fit_transform(df['gender'].astype(str))
    meal_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    meal_encoded = meal_ohe.fit_transform(df[['meal_category']])
    meal_df = pd.DataFrame(meal_encoded, columns=meal_ohe.get_feature_names_out(['meal_category']))
    features = pd.concat([df[['age','gender_enc']].reset_index(drop=True), meal_df.reset_index(drop=True)], axis=1)
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
        rest_pop_col.insert_one({'restaurant_id': str(rid), 'top_categories': counts.index.tolist(), 'counts': counts.tolist()})
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
    return {'trained_samples': int(X.shape[0])}

def predict_sentiment(texts):
    X = tfidf.transform(pd.Series(texts).astype(str))
    return sent_clf.predict_proba(X)[:,1]

def fit_cf(df):
    global svd, user_index, item_index, item_reverse_index
    t = df.groupby(['user_id','meal_category']).size().reset_index(name='rating')
    reader = Reader(rating_scale=(1, max(5, t['rating'].max())))
    data = Dataset.load_from_df(t[['user_id','meal_category','rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=20, random_state=42)
    svd.fit(trainset)
    user_index = {uid:i for i,uid in enumerate(t['user_id'].unique())}
    items = t['meal_category'].unique().tolist()
    item_index = {it:i for i,it in enumerate(items)}
    item_reverse_index = {i:it for it,i in item_index.items()}
    return {'user_count': len(user_index), 'item_count': len(item_index)}

def rank_with_cf(user_id, candidates):
    if svd is None or len(candidates)==0: return candidates
    scores = []
    for it in candidates:
        try: est = svd.predict(str(user_id), str(it)).est
        except: est = 0.0
        scores.append((it, est))
    # soft random shuffle by adding noise
    scores = [(c, s + random.uniform(-0.1, 0.1)) for c, s in scores]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores]

def build_sentiment_category_scores(df):
    if 'review' not in df.columns: return {}
    df2 = df[['meal_category','review']].dropna()
    if df2.empty: return {}
    probs = predict_sentiment(df2['review'])
    s = pd.DataFrame({'meal_category': df2['meal_category'].values, 'pos': probs})
    return s.groupby('meal_category')['pos'].mean().to_dict()

### ---- Helpers ----
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
    app.run(debug=True)
