import streamlit as st
import pandas as pd
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    model = joblib.load('models/voting_classifier_pipeline_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    logger.info(f"Loaded model with feature names: {feature_names}")
    # Print the type of the first preprocessor if it exists
    if hasattr(model, 'steps') and len(model.steps) > 0:
        logger.info(f"First pipeline step: {model.steps[0][0]}")
        if hasattr(model.steps[0][1], 'transformers'):
            logger.info("Preprocessor transformers:")
            for name, transformer, columns in model.steps[0][1].transformers:
                logger.info(f"- {name}: {transformer} (columns: {columns})")
    return model, feature_names

def predict_cancellation(input_data, model, feature_names):
    """Make prediction with proper feature handling."""
    try:
        logger.info(f"Expected features: {feature_names}")
        logger.info(f"Provided features: {list(input_data.keys())}")

        # Resolve meal_plan (multiselect -> single categorical value)
        meal_sel = input_data.get('meal_plan', [])
        if not isinstance(meal_sel, list):
            meal_sel = [meal_sel] if meal_sel else []
        # If user selected nothing, treat as 'Not Selected'
        if len(meal_sel) == 0:
            meal_plan_value = "Not Selected"
        elif len(meal_sel) == 1:
            meal_plan_value = meal_sel[0]
        else:
            # Should be blocked by UI/validation, but guard here
            raise ValueError("Select only one meal plan or none (Not Selected).")

        # Resolve market segment (selectbox -> single string)
        market_value = input_data.get('market_segment', "Other")

        clean_data = {
            "type_of_meal_plan": meal_plan_value,
            "market_segment_type": market_value
        }

        # Numeric features - ensure float and values are within expected range
        numeric_features = [
            'lead_time', 'avg_price_per_room', 'no_of_special_requests',
            'no_of_people', 'no_of_week_days'
        ]
        for feature in numeric_features:
            clean_data[feature] = float(input_data.get(feature, 0.0))

        # Build DataFrame and keep column order
        input_df = pd.DataFrame([clean_data])
        input_df = input_df[feature_names]

        logger.info("Clean input data:")
        logger.info(input_df.to_dict(orient='records')[0])

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        return prediction[0], probability[0]

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(f"Clean input data: {locals().get('clean_data',{})}")
        logger.error(f"Input DataFrame:\n{locals().get('input_df','N/A')}")
        raise

def validate_inputs(input_data, feature_names):
    """Validate that all required inputs are provided and within limits."""
    logger.info("Starting input validation...")

    # Numeric ranges required by spec
    ranges = {
        'lead_time': (0, 500),
        'avg_price_per_room': (20, 500),
        'no_of_special_requests': (0, 10),
        'no_of_people': (1, 5),
        'no_of_week_days': (0, 5)
    }

    # Convert and validate numeric fields
    for field, (min_v, max_v) in ranges.items():
        if field not in input_data:
            return False, f"Missing value for {field.replace('_',' ').title()}"
        try:
            val = float(input_data[field])
        except (TypeError, ValueError):
            return False, f"Invalid value for {field.replace('_',' ').title()}"
        if not (min_v <= val <= max_v):
            return False, f"{field.replace('_',' ').title()} must be between {min_v} and {max_v}"
        input_data[field] = val

    # Market segment must be selected (single choice from provided options)
    market_val = input_data.get('market_segment')
    if not market_val:
        return False, "Please select a Market Segment Type"
    if market_val not in ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation']:
        return False, "Invalid Market Segment selected"

    # Meal plan multiselect logic:
    meal_sel = input_data.get('meal_plan', [])
    if not isinstance(meal_sel, list):
        return False, "Meal Plan selection invalid"
    if len(meal_sel) > 1:
        return False, "Please select only one Meal Plan or none (Not Selected)."
    if len(meal_sel) == 1 and meal_sel[0] not in ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']:
        return False, "Invalid Meal Plan selected"

    logger.info("Input validation successful")
    return True, ""

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

        # First row: use sliders to constrain inputs and avoid accidental scroll changes
        c1, gap, c2 = st.columns([1, 0.2, 1])
        with c1:
            input_data['lead_time'] = st.slider(
                'Lead Time (days)',
                min_value=0, max_value=500, value=7,
                help="Days between booking and arrival (0–500). Use the slider to avoid overshoot."
            )
            st.caption("Allowed range: 0 — 500")

            input_data['avg_price_per_room'] = st.slider(
                'Average Price per Room (€)',
                min_value=20, max_value=500, value=75,
                help="Average price per night (20–500 €). Use the slider to avoid overshoot."
            )
            st.caption("Allowed range: 20 — 500 €")
        with c2:
            input_data['no_of_week_days'] = st.slider(
                'Number of Week Days',
                min_value=0, max_value=5, value=1,
                help="Number of week nights (0–5). Use the slider to avoid overshoot."
            )
            st.caption("Allowed range: 0 — 5")

            input_data['no_of_people'] = st.slider(
                'Number of People',
                min_value=1, max_value=5, value=2,
                help="Total number of adults and children (1–5). Use the slider to avoid overshoot."
            )
            st.caption("Allowed range: 1 — 5")

        st.markdown("### Additional Information")

        # Second row: meal plan multiselect + market segment selectbox
        c3, gap2, c4 = st.columns([1, 0.2, 1])
        with c3:
            input_data['no_of_special_requests'] = st.slider(
                'Number of Special Requests',
                min_value=0, max_value=10, value=0,
                help="Total special requests (0–10). Use the slider to avoid overshoot."
            )
            st.caption("Allowed range: 0 — 10")

            # multiselect (checklist). Allow zero selection -> interpret as "Not Selected".
            input_data['meal_plan'] = st.multiselect(
                'Select Meal Plan (choose at most one)',
                options=['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
                default=[]
            )
            if len(input_data['meal_plan']) > 1:
                st.error("Please select only one Meal Plan or none (Not Selected).")
        with c4:
            input_data['market_segment'] = st.selectbox(
                'Select Market Segment',
                options=['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'],
                index=0,
                help="Choose the booking market segment (single selection)."
            )
            st.caption("Single selection required")

    # Centered prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button('Predict Cancellation Probability')

    if predict_button:
        is_valid, error_message = validate_inputs(input_data, feature_names)
        if not is_valid:
            st.error(f"⚠️ {error_message}")
        else:
            try:
                # Ensure all model features exist (fill numeric defaults; categorical handled in predict)
                for feature in feature_names:
                    if feature not in input_data:
                        input_data[feature] = 0.0

                prediction, probability = predict_cancellation(input_data, model, feature_names)

                # Display results (unchanged)
                st.markdown("### Prediction Results")
                if prediction == 1:
                    st.markdown(
                        f"""<div class="prediction-box warning-box">
                            <h2>⚠️ High Risk of Cancellation</h2>
                            <h3>Cancellation Probability: {probability[1]:.1%}</h3>
                        </div>""", unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<div class="prediction-box success-box">
                            <h2>✅ Low Risk of Cancellation</h2>
                            <h3>Booking Stability: {probability[0]:.1%}</h3>
                        </div>""", unsafe_allow_html=True
                    )

                st.markdown("### Detailed Probabilities")
                c_a, c_b = st.columns(2)
                with c_a:
                    st.metric("Probability of Keeping Reservation", f"{probability[0]:.1%}")
                with c_b:
                    st.metric("Probability of Cancellation", f"{probability[1]:.1%}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.error("Please ensure all input values are valid and try again.")

if __name__ == '__main__':
    main()
