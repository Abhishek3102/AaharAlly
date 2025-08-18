# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import pandas as pd
# import numpy as np

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# def process_data():
#     # Load the data from the CSV file
#     df = pd.read_csv("final_updated_dataset.csv")

#     # Define the age bins and labels
#     bins = [0, 20, 25, 35, 45, 60, 100] 
#     labels = ['<20', '20-25', '25-35', '35-45', '45-60', '60+']
    
#     # Create age groups
#     df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

#     # Group data by age_group and meal_category
#     age_food_group_counts = df.groupby(['age_group', 'meal_category']).size().unstack(fill_value=0)

#     # Determine the most ordered food for each age group
#     most_ordered_per_age_group = age_food_group_counts.idxmax(axis=1)
#     most_ordered_counts = age_food_group_counts.max(axis=1)

#     # Create a result DataFrame
#     result = pd.DataFrame({
#         'food_category': most_ordered_per_age_group,
#         'no_of_people': most_ordered_counts
#     })
    
#     result['Age Range'] = ' (' + result.index.astype(str) + ')'
    
#     return result.to_dict(orient='records')

# @app.route('/api/get_data', methods=['GET'])
# def get_data():
#     try:
#         data = process_data()
#         return jsonify({'success': True, 'data': data})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# if __name__ == '__main__':  # Corrected line
#     app.run(debug=True)


from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def process_data():
    # Load the data from the CSV file
    df = pd.read_csv("final_updated_dataset.csv")

    # Define the age bins and labels
    bins = [0, 20, 25, 35, 45, 60, 100] 
    labels = ['<20', '20-25', '25-35', '35-45', '45-60', '60+']
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # Group data by age_group and category
    age_category_counts = df.groupby(['age_group', 'category']).size().unstack(fill_value=0)

    # Determine the most purchased category for each age group
    most_ordered_per_age_group = age_category_counts.idxmax(axis=1)
    most_ordered_counts = age_category_counts.max(axis=1)

    # Create a result DataFrame
    result = pd.DataFrame({
        'category': most_ordered_per_age_group,
        'no_of_people': most_ordered_counts
    })
    
    result['Age Range'] = ' (' + result.index.astype(str) + ')'
    
    return result.to_dict(orient='records')

@app.route('/api/get_data', methods=['GET'])
def get_data():
    try:
        data = process_data()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
