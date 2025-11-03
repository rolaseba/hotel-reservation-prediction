import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; padding: 1rem; }
        .sub-header { font-size: 1.5rem; color: #424242; padding-bottom: 1rem; }
        .stButton > button { background-color: #1E88E5; color: white; padding: 0.5rem 2rem; font-size: 1.2rem; border-radius: 0.5rem; border: none; width: 100%; margin-top: 2rem; }
        .prediction-box { padding: 1.5rem; border-radius: 0.5rem; margin-top: 1rem; text-align: center; }
        .success-box { background-color: #4CAF50; color: white; }
        .warning-box { background-color: #f44336; color: white; }

        /* Limit width of input widgets */
        .stNumberInput, .stTextInput, .stSelectbox, .stCheckbox {
            max-width: 360px;
        }
        /* Tighter visual spacing for inner columns */
        .block-container .css-1lcbmhc { padding-top: 0.25rem; }
    </style>
""", unsafe_allow_html=True)

# Load the saved model and feature names
@st.cache_resource
def load_model():
    model = joblib.load('models/voting_classifier_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

def predict_cancellation(input_data, model, feature_names):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return prediction[0], probability[0]

def main():
    # Header
    st.markdown('<h1 class="main-header">Hotel Reservation Cancellation Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter booking details to predict cancellation probability</p>', unsafe_allow_html=True)

    # Load model
    model, feature_names = load_model()

    # Use a centered narrow column for inputs
    left, center, right = st.columns([1, 0.6, 1])
    with center:
        st.markdown("### Booking Details")
        input_data = {}
        # add a small gap column between the two input columns
        c1, gap, c2 = st.columns([1, 0.2, 1])
        with c1:
            input_data['lead_time'] = st.number_input('Lead Time (days)', min_value=0, help="Number of days between the date of booking and the arrival date")
            input_data['avg_price_per_room'] = st.number_input('Average Price per Room (€)', min_value=0.0, help="Average price per day of the reservation (in euros); prices are dynamic")
        with c2:
            input_data['no_of_week_days'] = st.number_input('Number of Week Days', min_value=0, help="Number of weeks the guest stayed or booked to stay at the hotel")
            input_data['no_of_people'] = st.number_input('Number of People', min_value=1, help="Total number of adults and kids")

        st.markdown("### Additional Information")
        # add a small gap column here too
        c3, gap2, c4 = st.columns([1, 0.2, 1])
        with c3:
            input_data['no_of_special_requests'] = st.number_input('Number of Special Requests', min_value=0, help="Total number of special requests made by the customer (e.g., high floor, view from the room, etc.)")
            input_data['type_of_meal_plan_Meal Plan 1'] = st.checkbox('Meal Plan 1 booked by the customer')
        with c4:
            input_data['type_of_meal_plan_Meal Plan 2'] = st.checkbox('Meal Plan 2 booked by the customer')
            input_data['market_segment_type_Corporate'] = st.checkbox('Corporate Booking')
            input_data['market_segment_type_Online'] = st.checkbox('Online Booking')

    # Centered prediction button (unchanged behavior, but ensure it's centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button('Predict Cancellation Probability')

    if predict_button:
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_data:
                input_data[feature] = 0

        # Get prediction
        prediction, probability = predict_cancellation(input_data, model, feature_names)

        # Display results
        st.markdown("### Prediction Results")
        
        if prediction == 1:
            st.markdown(
                f"""
                <div class="prediction-box warning-box">
                    <h2>⚠️ High Risk of Cancellation</h2>
                    <h3>Cancellation Probability: {probability[1]:.1%}</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-box success-box">
                    <h2>✅ Low Risk of Cancellation</h2>
                    <h3>Booking Stability: {probability[0]:.1%}</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Show detailed probabilities
        st.markdown("### Detailed Probabilities")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of Keeping Reservation", f"{probability[0]:.1%}")
        with col2:
            st.metric("Probability of Cancellation", f"{probability[1]:.1%}")

if __name__ == '__main__':
    main()
