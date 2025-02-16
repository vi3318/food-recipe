import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("recipe_final.csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'].fillna(""))

# Train KNN Model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_ingredients)

def recommend_recipes(ingredients):
    input_ingredients_transformed = vectorizer.transform([ingredients])
    distances, indices = knn.kneighbors(input_ingredients_transformed)
    recommendations = data.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'calories', 'image_url']].head(5)

# Streamlit UI
st.title("🍽️ Smart Recipe Recommender")
st.markdown("Enter your available ingredients, and we'll suggest the best recipes with estimated calories!")

# User Input
ingredients = st.text_area("Enter ingredients (comma separated)")

if st.button("Find Recipes"):
    if ingredients.strip():
        recommendations = recommend_recipes(ingredients)
        for _, row in recommendations.iterrows():
            st.subheader(row['recipe_name'])
            st.image(row['image_url'], width=300, caption=f"Calories: {row['calories']}")
            st.write(f"**Ingredients:** {row['ingredients_list']}")
            st.markdown("---")
    else:
        st.warning("Please enter some ingredients to get recommendations!")