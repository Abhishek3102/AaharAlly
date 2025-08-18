from flask import Flask, jsonify
import pandas as pd
from textblob import TextBlob
import os

app = Flask(__name__)

# Function for sentiment analysis on English reviews
def analyze_sentiment(review):
    try:
        # Perform sentiment analysis using TextBlob
        sentiment_polarity = TextBlob(review).sentiment.polarity

        # Classify as Positive, Neutral, or Negative based on polarity value
        if sentiment_polarity > 0:
            return "Positive"
        elif sentiment_polarity == 0:
            return "Neutral"
        else:
            return "Negative"
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return "Error in sentiment analysis"

# Function to load reviews from CSV and perform sentiment analysis
def analyze_reviews_from_csv(file_path="final_updated_dataset (1).csv"):
    try:
        # Get the absolute path to the CSV file
        abs_file_path = os.path.join(os.path.dirname(__file__), file_path)
        
        # Load the CSV file containing the reviews
        df = pd.read_csv(abs_file_path)

        # Ensure the 'review' column exists
        if 'review' not in df.columns:
            raise Exception("CSV file does not contain a 'review' column.")

        # Apply sentiment analysis on the 'review' column
        df['sentiment'] = df['review'].apply(analyze_sentiment)

        # Convert the results to a list of dictionaries for the JSON response
        sentiment_results = df[['review', 'sentiment']].to_dict(orient='records')

        return sentiment_results
    except FileNotFoundError:
        print(f"CSV file not found at {abs_file_path}.")
        return None
    except pd.errors.EmptyDataError:
        print("CSV file is empty.")
        return None
    except Exception as e:
        print(f"Error processing the file: {e}")
        return None

# Flask route for sentiment analysis from CSV
@app.route('/api/sentiment', methods=['GET'])
def sentiment_analysis():
    try:
        # Perform sentiment analysis on the reviews from the CSV file
        sentiment_results = analyze_reviews_from_csv()

        if sentiment_results:
            return jsonify({'success': True, 'data': sentiment_results})
        else:
            return jsonify({'success': False, 'message': 'Error analyzing the reviews'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
