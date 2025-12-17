from flask import Flask, jsonify
import pandas as pd
from textblob import TextBlob
from googletrans import Translator
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# MongoDB Client setup
client = MongoClient(os.getenv("MONGO_URL"))
db = client['aahar_ally']
collection = db['cluster']

# Helper function to convert ObjectId to string
def convert_objectid(data):
    if isinstance(data, list):
        for doc in data:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
    elif isinstance(data, dict):
        if '_id' in data:
            data['_id'] = str(data['_id'])
    return data

# Function for sentiment analysis
def translate_and_analyze_sentiment(review):
    translator = Translator()
    try:
        # Detect the language of the review
        detected_language = translator.detect(review).lang

        # Translate the review to English if it's not already in English
        if detected_language != 'en':
            translated_review = translator.translate(review, dest='en').text
        else:
            translated_review = review

        # Perform sentiment analysis on the translated text
        sentiment_polarity = TextBlob(translated_review).sentiment.polarity

        # Classify as Positive, Neutral, or Negative based on polarity value
        if sentiment_polarity > 0:
            return "Positive"
        elif sentiment_polarity == 0:
            return "Neutral"
        else:
            return "Negative"
    except Exception as e:
        # Handle errors in detection or translation
        return "Error in sentiment analysis"

# Flask route for sentiment analysis
@app.route('/api/sentiment_analysis', methods=['GET'])
def sentiment_analysis():
    try:
        # Load reviews from CSV
        df = pd.read_csv("MOCK_DATA (4).csv")  
        df['sentiment'] = df['review'].apply(translate_and_analyze_sentiment)

        sentiment_results = df[['review', 'sentiment']].to_dict(orient='records')
        return jsonify({'success': True, 'data': sentiment_results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_clustering_data():
    try:
        
        df = pd.read_csv("MOCK_DATA (4).csv")  

        bins = [0, 20, 25, 35, 45, 60, 100]
        labels = ['<20', '20-25', '25-35', '35-45', '45-60', '60+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

        # Grouping data by age and food category
        age_food_group_counts = df.groupby(['age_group', 'meal_category']).size().unstack(fill_value=0)

        # Finding the most ordered dish per age group
        most_ordered_per_age_group = age_food_group_counts.idxmax(axis=1)
        most_ordered_counts = age_food_group_counts.max(axis=1)

        # Creating the result DataFrame
        result = pd.DataFrame({
            'Age Range': most_ordered_per_age_group.index.astype(str),
            'Most Ordered Dish': most_ordered_per_age_group,
            'Number of People': most_ordered_counts,
            'Food Category': most_ordered_per_age_group  # Assuming this is the category of the food item
        })

        return result.to_dict(orient='records')

    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Flask route to fetch and store processed data (Clustering)
@app.route('/api/clustering', methods=['GET'])
def clustering():
    try:
        # Process the data from CSV
        data = process_clustering_data()

        if data is not None:
            # Store the processed data in MongoDB
            collection.delete_many({})  # Clears previous data
            inserted_ids = collection.insert_many(data)
            if inserted_ids:
                # Convert ObjectId to string for JSON serialization
                data_with_ids = list(collection.find())
                data_with_ids = convert_objectid(data_with_ids)

                return jsonify({'success': True, 'message': 'Data stored successfully', 'data': data_with_ids})
            else:
                return jsonify({'success': False, 'message': 'Error storing data in MongoDB'})
        else:
            return jsonify({'success': False, 'message': 'Error processing data'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
