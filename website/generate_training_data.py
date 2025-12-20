
import pandas as pd
import random
import numpy as np

# 1. Master Category List (Aligned with Frontend/DB)
categories = [
    "Cheesy", "Chinese", "Dessert", "Healthy", "Indian Curry", 
    "Seafood", "Snacks", "Spicy", "Street Food", "Vegan", 
    "Biryani", "Pizza", "Burger", "South Indian" # Added potential future categories
]

# 2. Complex Demographic Patterns
# The ML learns these probabilities. 
# Format: {AgeRange: (Min, Max), Gender: Str, Preferences: {Category: Weight}}
profiles = [
    # Young Adults (Male) - Love Fast Food, Spicy
    {
        "name": "Gen Z Male",
        "age": (15, 24),
        "gender": "Male",
        "weights": {"Street Food": 30, "Spicy": 25, "Cheesy": 20, "Snacks": 15, "Chinese": 10, "Healthy": 1}
    },
    # Young Adults (Female) - Love Dessert, Vegan, Street Food
    {
        "name": "Gen Z Female",
        "age": (15, 24),
        "gender": "Female",
        "weights": {"Dessert": 30, "Vegan": 20, "Street Food": 20, "Chinese": 15, "Cheesy": 10, "Spicy": 5}
    },
    # Working Professionals (Male) - Curry, Seafood, Chinese
    {
        "name": "Adult Male",
        "age": (25, 45),
        "gender": "Male",
        "weights": {"Indian Curry": 30, "Seafood": 20, "Chinese": 20, "Biryani": 15, "Spicy": 10, "Healthy": 5}
    },
    # Working Professionals (Female) - Healthy, Vegan, Chinese
    {
        "name": "Adult Female",
        "age": (25, 45),
        "gender": "Female",
        "weights": {"Healthy": 35, "Vegan": 25, "Chinese": 15, "Dessert": 15, "Indian Curry": 10}
    },
    # Seniors (Male) - Traditional, Healthy
    {
        "name": "Senior Male",
        "age": (46, 80),
        "gender": "Male",
        "weights": {"Indian Curry": 40, "Healthy": 30, "South Indian": 20, "Seafood": 10}
    },
    # Seniors (Female) - Traditional, Healthy, Vegan
    {
        "name": "Senior Female",
        "age": (46, 80),
        "gender": "Female",
        "weights": {"Healthy": 35, "Indian Curry": 30, "Vegan": 20, "South Indian": 15}
    }
]

# 3. Sentiment Templates
positive_templates = [
    "I loved the {cat}!", "The {cat} was amazing.", "Best {cat} I've ever had.",
    "Highly recommend the {cat}.", "Delicious {cat}, will order again.",
    "Perfect flavor in this {cat}.", "A delightful {cat} experience."
]
negative_templates = [
    "The {cat} was terrible.", "Worst {cat} ever.", "Do not order the {cat}.",
    "Disappointed with the {cat}.", "The {cat} was cold and tasteless.",
    "Too salty, hated the {cat}.", "Bad experience with {cat}."
]

data = []
user_counter = 2000 # High ID range

# 4. Generate 2000 Rows of Rich Data
total_users_to_sim = 800

for _ in range(total_users_to_sim):
    # Pick a random profile based on population distribution
    profile = random.choice(profiles)
    
    uid = f"U{user_counter}"
    age = random.randint(profile['age'][0], profile['age'][1])
    gender = profile['gender']
    
    # User makes 1-5 interactions
    num_orders = random.randint(1, 5)
    
    # Calculate probabilities for this user
    cats = list(profile['weights'].keys())
    wts = list(profile['weights'].values())
    total_w = sum(wts)
    norm_wts = [w/total_w for w in wts]
    
    for _ in range(num_orders):
        # Pick category based on profile weights
        try:
            chosen_cat = np.random.choice(cats, p=norm_wts)
        except:
            chosen_cat = random.choice(categories) # Fallback
            
        # Determine sentiment (Mostly positive if it matches preference, but sometimes bad luck)
        is_positive = random.random() < 0.85 
        
        if is_positive:
            tpl = random.choice(positive_templates)
        else:
            tpl = random.choice(negative_templates)
            
        review = tpl.format(cat=chosen_cat)
        
        # Add to dataset
        data.append([uid, "AaharAlly_Main", age, gender, chosen_cat, review])
        
    user_counter += 1

# 5. Save
df = pd.DataFrame(data, columns=["user_id", "restaurant_id", "age", "gender", "meal_category", "review"])
output_path = "c:/aaharally/AaharAlly/ML_Service/train_data.csv"
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} rows of Demographic-Aware Data at {output_path}")
print("Sample:")
print(df.head())
