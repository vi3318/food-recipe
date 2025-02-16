import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import base64

# Set page title and layout
st.set_page_config(page_title="Vidarsh - AI Cooking Assistant", layout="wide")

# Load dataset
data = pd.read_csv("recipe_final.csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'].fillna(""))

# Train KNN Model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X_ingredients)

def recommend_recipes(ingredients):
    input_ingredients_transformed = vectorizer.transform([ingredients])
    distances, indices = knn.kneighbors(input_ingredients_transformed)
    recommendations = data.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'calories', 'image_url']]

# Sidebar for PDF Report
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.sidebar.title("Navigation")
if st.sidebar.button("📄 View Project Docs"):
    display_pdf("project_report.pdf")

# Main UI
st.title("🍽️ Vidarsh - AI Cooking Assistant")
st.markdown("Helping you find the best recipes based on ingredients!")

# User Input
ingredients_input = st.text_area("Enter ingredients you have (comma-separated):")

if st.button("🔍 Find Recipes"):
    if ingredients_input.strip():
        recommendations = recommend_recipes(ingredients_input)
        st.subheader("🍽️ Recommended Recipes")
        for _, row in recommendations.iterrows():
            st.markdown(f"### {row['recipe_name']}")
            st.write(f"**Calories:** {row['calories']}")
            st.write(f"**Ingredients:** {row['ingredients_list']}")
            if pd.notna(row['image_url']):
                st.image(row['image_url'], width=300)
            st.markdown("---")
    else:
        st.warning("Please enter some ingredients!")
