from flask import Flask, jsonify
import pandas as pd
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


# Function to process data
def process_data():
    try:
        # Load dataset
        df = pd.read_csv("MOCK_DATA (4).csv")

        # Binning ages into groups
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
            'age_range': most_ordered_per_age_group.index.astype(str),
            'no_of_people': most_ordered_counts,
            'food_category': most_ordered_per_age_group
        })

        return result.to_dict(orient='records')

    except Exception as e:
        print(f"Error processing data: {e}")
        return None


# Function to store data in MongoDB
def store_data_in_mongo(data):
    try:
        collection.delete_many({})  # Clears previous data
        inserted_ids = collection.insert_many(data)
        return inserted_ids
    except Exception as e:
        print(f"Error storing data in MongoDB: {e}")
        return None


# Flask route to fetch and store processed data
@app.route('/api/get_data', methods=['GET'])
def get_data():
    try:
        # Process the data from CSV
        data = process_data()

        if data is not None:
            # Store the processed data in MongoDB
            inserted_ids = store_data_in_mongo(data)
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
