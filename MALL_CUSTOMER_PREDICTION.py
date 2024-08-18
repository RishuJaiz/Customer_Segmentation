import streamlit as st
import numpy as np
import pickle

# Function to load the model
def load_model():
    with open('finalized_model.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Main function to render the Streamlit UI
def main():
    st.title('Customer Segmentation Prediction Interface')
    st.write('This application predicts customer segments based on their age, annual income, and spending score.')

    # User inputs for the model
    age = st.number_input('Enter Age', min_value=18, max_value=100, value=30, step=1)
    annual_income = st.number_input('Enter Annual Income (k$)', min_value=15, max_value=150, value=50, step=1)
    spending_score = st.number_input('Enter Spending Score (1-100)', min_value=1, max_value=100, value=50, step=1)

    # Button to make prediction
    if st.button('Predict'):
        # Preparing the features as per the model's requirement
        features = np.array([[age, annual_income, spending_score]])
        # Standardizing the features (ensure to load or define the scaler used during model training)
        # Example: scaler = load_scaler() or define the scaling logic here
        # features_scaled = scaler.transform(features)

        # Direct prediction without scaling for demonstration
        prediction = model.predict(features)
        st.success(f'The predicted customer segment is: {prediction[0]}')

if __name__ == '__main__':
    main()
