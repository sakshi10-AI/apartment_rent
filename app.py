import streamlit as st
import pandas as pd
import pickle

# Set page title
st.set_page_config(page_title="Apartment Rent Prediction", page_icon="🏠")

# Title
st.title("🏠 Apartment Rent Prediction App")

# Sidebar: Model Selection
st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Select Model", ("Linear Regression", "Random Forest"))

# Load corresponding model
if model_choice == "Linear Regression":
    model = pickle.load(open('linear_regression_rent_model_pipeline.pkl', 'rb'))
else:
    model = pickle.load(open('random_forest_rent_model_pipeline.pkl', 'rb'))

# Sidebar: Apartment Input
st.sidebar.header("Enter Apartment Details")

def user_input_features():
    location = st.sidebar.selectbox("Location ID (Encoded)", [0,1,2,3,4,5,6,7,8,9])  # Update as needed
    bhk = st.sidebar.number_input("BHK (No. of Bedrooms)", min_value=1, max_value=10, value=2)
    floor = st.sidebar.number_input("Floor Number", min_value=0, max_value=100, value=2)
    furnished = st.sidebar.selectbox("Furnished?", ("No", "Yes"))
    area_sqft = st.sidebar.number_input("Area (sqft)", min_value=100, max_value=10000, value=1000)
    age_of_building = st.sidebar.number_input("Age of Building (Years)", min_value=0, max_value=100, value=5)

    # Convert 'furnished' to numeric (Yes=1, No=0)
    furnished = 1 if furnished == 'Yes' else 0

    # Create DataFrame
    data = {
        'location': location,
        'bhk': bhk,
        'floor': floor,
        'furnished': furnished,
        'area_sqft': area_sqft,
        'age_of_building': age_of_building
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Force correct column order
feature_order = ['location', 'bhk', 'floor', 'furnished', 'area_sqft', 'age_of_building']
input_df = input_df[feature_order]

# Main Panel: Display input
st.subheader('Apartment Details Entered:')
st.write(input_df)

# Predict Button
if st.button('Predict Rent'):
    prediction = model.predict(input_df)
    st.subheader('Predicted Monthly Rent:')
    st.success(f"₹ {prediction[0]:,.2f}")

# Footer
st.caption("Made with ❤️ using Streamlit")
