from flask import Flask, request, jsonify
import pandas as pd
from textblob import TextBlob
from googletrans import Translator
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__)

# MongoDB Client setup
client = MongoClient('mongodb+srv://maurya48ashish:Ashish48Maurya@cluster0.w5ltbks.mongodb.net/')
db = client['aahar_ally']
orders_collection = db['orders']          # stores user order history
clusters_collection = db['clusters']      # stores clustering results
reviews_collection = db['reviews']        # stores sentiment results

# convert ObjectId to string
def convert_objectid(data):
    if isinstance(data, list):
        for doc in data:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
    elif isinstance(data, dict):
        if '_id' in data:
            data['_id'] = str(data['_id'])
    return data

# --- Sentiment Analysis ---
def translate_and_analyze_sentiment(review):
    translator = Translator()
    try:
        detected_language = translator.detect(review).lang
        translated_review = translator.translate(review, dest='en').text if detected_language != 'en' else review
        sentiment_polarity = TextBlob(translated_review).sentiment.polarity
        if sentiment_polarity > 0:
            return "Positive"
        elif sentiment_polarity == 0:
            return "Neutral"
        else:
            return "Negative"
    except:
        return "Error in sentiment analysis"

@app.route('/api/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    try:
        data = request.get_json()
        review = data.get("review")
        user_id = data.get("user_id")
        sentiment = translate_and_analyze_sentiment(review)

        # Store in MongoDB
        reviews_collection.insert_one({
            "user_id": user_id,
            "review": review,
            "sentiment": sentiment
        })

        return jsonify({'success': True, 'review': review, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# --- Clustering & Recommendation ---
def process_clustering_data():
    try:
        df = pd.read_csv("MOCK_DATA (4).csv")

        # Age bins
        bins = [0, 20, 25, 35, 45, 60, 100]
        labels = ['<20', '20-25', '25-35', '35-45', '45-60', '60+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        # Group by age_group & meal_category
        age_food_group_counts = df.groupby(['age_group', 'meal_category']).size().unstack(fill_value=0)

        # Most ordered dish per age group
        most_ordered_per_age_group = age_food_group_counts.idxmax(axis=1)
        most_ordered_counts = age_food_group_counts.max(axis=1)

        result = pd.DataFrame({
            'Age Range': most_ordered_per_age_group.index.astype(str),
            'Most Ordered Dish': most_ordered_per_age_group,
            'Number of People': most_ordered_counts,
            'Food Category': most_ordered_per_age_group
        })

        return result.to_dict(orient='records')
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


@app.route('/api/clustering', methods=['GET'])
def clustering():
    try:
        data = process_clustering_data()
        if data is not None:
            clusters_collection.delete_many({})
            clusters_collection.insert_many(data)
            return jsonify({'success': True, 'message': 'Cluster data stored', 'data': convert_objectid(data)})
        else:
            return jsonify({'success': False, 'message': 'Error in clustering'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# --- Recommendation API ---
@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        age = data.get("age")
        gender = data.get("gender")
        restaurant_id = data.get("restaurant_id")

        # Check if user exists in order history
        user_history = list(orders_collection.find({"user_id": user_id}))
        is_new_user = len(user_history) == 0

        # Step 1: Get demographic cluster recommendation
        bins = [0, 20, 25, 35, 45, 60, 100]
        labels = ['<20', '20-25', '25-35', '35-45', '45-60', '60+']
        age_group = pd.cut([age], bins=bins, labels=labels, right=False)[0]
        cluster_data = clusters_collection.find_one({"Age Range": str(age_group)})
        cluster_recommend = cluster_data['Most Ordered Dish'] if cluster_data else None

        # Step 2: If new user → return demographic + restaurant popular dish
        if is_new_user:
            return jsonify({
                'success': True,
                'user_type': 'new',
                'age_group': str(age_group),
                'recommendations': [cluster_recommend]
            })

        # Step 3: If returning user → filter by past categories
        past_categories = [order['meal_category'] for order in user_history]
        # Narrow recommendation pool
        personalized_recs = list(set([cluster_recommend] + past_categories))

        return jsonify({
            'success': True,
            'user_type': 'returning',
            'age_group': str(age_group),
            'recommendations': personalized_recs
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# --- Store Order API (to save history) ---
@app.route('/api/store_order', methods=['POST'])
def store_order():
    try:
        data = request.get_json()
        orders_collection.insert_one(data)
        return jsonify({'success': True, 'message': 'Order stored'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
