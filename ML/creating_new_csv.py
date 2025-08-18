import pandas as pd
import random

# Load your original dataset
original_path = 'MOCK_DATA-_4_ (1).csv'  # Change this to your actual path
df = pd.read_csv(original_path)

# Generate synthetic user_id and restaurant_id
df['user_id'] = ['U' + str(i + 1) for i in range(len(df))]
df['restaurant_id'] = ['R' + str(i + 1) for i in range(len(df))]

# Assign random genders
df['gender'] = [random.choice(['Male', 'Female']) for _ in range(len(df))]

# Reorder and select required columns
required_columns = ['user_id', 'restaurant_id', 'age', 'gender', 'meal_category', 'review']
df_out = df[required_columns]

# Save to new CSV
output_path = 'train_data.csv'
df_out.to_csv(output_path, index=False)

print(f"✅ New dataset saved to: {output_path}")
