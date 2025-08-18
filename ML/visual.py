import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Sentiment & Segmentation Analysis", layout="wide")

# Sentiment Analysis Function
def analyze_sentiment(review):
    try:
        sentiment_polarity = TextBlob(review).sentiment.polarity
        if sentiment_polarity > 0:
            return "Positive"
        elif sentiment_polarity == 0:
            return "Neutral"
        else:
            return "Negative"
    except Exception as e:
        return "Error"

# Load and preprocess data
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['sentiment'] = df['review'].apply(analyze_sentiment)
    
    # Define age bins and labels
    bins = [0, 15, 25, 35, 45, 50, 100]
    labels = ['<20', '20-25', '25-35', '35-45', '45-50', '50+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df

# Visualization functions
def plot_sentiment_distribution(data):
    st.subheader("Sentiment Distribution")
    sentiment_counts = data['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['green', 'gray', 'red']
    )
    ax.set_title("Sentiment Analysis Results")
    ax.legend(sentiment_counts.index, title="Sentiment", loc="upper right")
    st.pyplot(fig)

def plot_category_by_age(data):
    st.subheader("Category Popularity by Age Group")
    age_category_counts = data.groupby(['age_group', 'category']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        age_category_counts,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=True,
        ax=ax
    )
    ax.set_title("Age Group vs. Product Category Popularity")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Age Group")
    st.pyplot(fig)

def plot_most_purchased_category(data):
    st.subheader("Most Purchased Category by Age Group")
    
    # Calculate most purchased categories and their counts by age group
    # age_category_counts = data.groupby(['age_group', 'category']).size().unstack(fill_value=0)
    age_category_counts = data.groupby(['age_group', 'meal_category']).size().unstack(fill_value=0)
    most_ordered_per_age_group = age_category_counts.idxmax(axis=1)
    most_ordered_counts = age_category_counts.max(axis=1)
    result = pd.DataFrame({
        'Age Range': ' (' + most_ordered_per_age_group.index.astype(str) + ')',
        'meal_category': most_ordered_per_age_group,
        # 'category': most_ordered_per_age_group,
        'no_of_people': most_ordered_counts
    })
    
    # Create the bar plot with individual labels
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        result['Age Range'], 
        result['no_of_people'], 
        color=plt.cm.tab10(range(len(result))),
        edgecolor="black"
    )
    
    # Add category labels above each bar
    # for bar, category in zip(bars, result['category']):
    for bar, meal_category in zip(bars, result['meal_category']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 0.5, 
            meal_category, 
            # category, 
            ha='center', 
            va='bottom', 
            fontsize=9
        )
    
    # Add title, labels, and legend
    ax.set_title("Most Purchased Categories per Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Number of People")
    ax.set_xticks(range(len(result)))
    ax.set_xticklabels(result['Age Range'], rotation=0)
    # ax.legend(bars, result['category'], title="Categories", loc="upper right")
    ax.legend(bars, result['meal_category'], title="Categories", loc="upper right")
    
    st.pyplot(fig)


# Main Streamlit App
def main():
    st.title("E-Commerce Sentiment & Segmentation Analysis")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        # Load and display data
        data = load_data(uploaded_file)
        st.dataframe(data.head())

        # Sidebar filters
        st.sidebar.header("Filters")
        sentiment_filter = st.sidebar.multiselect("Filter by Sentiment", options=data['sentiment'].unique(), default=data['sentiment'].unique())
        category_filter = st.sidebar.multiselect("Filter by Category", options=data['meal_category'].unique(), default=data['meal_category'].unique())
        # category_filter = st.sidebar.multiselect("Filter by Category", options=data['category'].unique(), default=data['category'].unique())
        age_group_filter = st.sidebar.multiselect("Filter by Age Group", options=data['age_group'].unique(), default=data['age_group'].unique())

        # Apply filters
        filtered_data = data[
            (data['sentiment'].isin(sentiment_filter)) &
            (data['meal_category'].isin(category_filter)) &
            # (data['category'].isin(category_filter)) &
            (data['age_group'].isin(age_group_filter))
        ]
        
        st.write(f"Filtered Data (Rows: {len(filtered_data)})")
        st.dataframe(filtered_data)

        # Visualizations
        st.header("Visualizations")
        plot_sentiment_distribution(filtered_data)
        plot_category_by_age(filtered_data)
        plot_most_purchased_category(filtered_data)

        # Download filtered data
        st.sidebar.header("Download Data")
        csv_data = filtered_data.to_csv(index=False)
        st.sidebar.download_button("Download Filtered Data as CSV", csv_data, "filtered_data.csv", "text/csv")
    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
