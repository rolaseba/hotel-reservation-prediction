import streamlit as st
import pandas as pd
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION PARAMETERS ==========
# Threshold for cancellation risk classification (0.0 to 1.0)
# Probabilities >= this threshold are considered "High Risk"
# Probabilities < this threshold are considered "Low Risk"
CANCELLATION_RISK_THRESHOLD = 0.6  # 60% threshold
# ============================================

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

        /* Make input labels bigger and bold */
        .stSlider > div > div > div > label { font-size: 2.5rem !important; font-weight: bold !important; }
        .stSelectbox > div > div > label { font-size: 2.5rem !important; font-weight: bold !important; }
        .stMultiselect > div > div > label { font-size: 2.5rem !important; font-weight: bold !important; }
    </style>
""", unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model():
    model = joblib.load('models/voting_classifier_pipeline_model.pkl')
    logger.info("Model loaded successfully with PCA pipeline")
    # Extract expected input features (before transformation)
    if hasattr(model, 'named_steps'):
        preprocessor = model.named_steps.get('preprocessor')
        if preprocessor:
            logger.info("Preprocessor found with transformers:")
            for name, transformer, columns in preprocessor.transformers:
                logger.info(f"- {name}: {columns}")
    return model

def predict_cancellation(input_data, model):
    """Make prediction with proper feature handling through the preprocessing pipeline."""
    try:
        # Map user input to original feature names expected by preprocessor
        clean_data = {
            "no_of_adults": int(input_data.get('no_of_adults', 0)),
            "no_of_children": int(input_data.get('no_of_children', 0)),
            "no_of_weekend_nights": int(input_data.get('no_of_weekend_nights', 0)),
            "no_of_week_nights": int(input_data.get('no_of_week_nights', 0)),
            "lead_time": float(input_data.get('lead_time', 0.0)),
            "avg_price_per_room": float(input_data.get('avg_price_per_room', 0.0)),
            "no_of_special_requests": int(input_data.get('no_of_special_requests', 0)),
            "type_of_meal_plan": input_data.get('meal_plan', "Not Selected"),
            "market_segment_type": input_data.get('market_segment', "Online")
        }

        # Build DataFrame with original feature names
        input_df = pd.DataFrame([clean_data])
        
        logger.info("Input data for preprocessing:")
        logger.info(input_df.to_dict(orient='records')[0])

        # The preprocessor in the pipeline will handle PCA transformation
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        logger.info(f"Prediction: {prediction[0]}, Probabilities: {probability[0]}")
        return prediction[0], probability[0]

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(f"Clean input data: {locals().get('clean_data', {})}")
        raise

def validate_inputs(input_data):
    """Validate that all required inputs are provided and within limits."""
    logger.info("Starting input validation...")

    # Numeric ranges required by spec
    ranges = {
        'lead_time': (0, 500),
        'avg_price_per_room': (20, 500),
        'no_of_special_requests': (0, 10),
        'no_of_adults': (0, 5),
        'no_of_children': (0, 5),
        'no_of_weekend_nights': (0, 7),
        'no_of_week_nights': (0, 30)
    }

    # Convert and validate numeric fields
    for field, (min_v, max_v) in ranges.items():
        if field not in input_data:
            return False, f"Missing value for {field.replace('_', ' ').title()}"
        try:
            val = float(input_data[field])
        except (TypeError, ValueError):
            return False, f"Invalid value for {field.replace('_', ' ').title()}"
        if not (min_v <= val <= max_v):
            return False, f"{field.replace('_', ' ').title()} must be between {min_v} and {max_v}"
        input_data[field] = val

    # Market segment must be selected (single choice from provided options)
    market_val = input_data.get('market_segment')
    if not market_val:
        return False, "Please select a Market Segment Type"
    if market_val not in ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation']:
        return False, "Invalid Market Segment selected"

    # Meal plan multiselect logic
    meal_sel = input_data.get('meal_plan', [])
    if not isinstance(meal_sel, list):
        return False, "Meal Plan selection invalid"
    if len(meal_sel) > 1:
        return False, "Please select only one Meal Plan or none (Not Selected)."
    if len(meal_sel) == 0:
        input_data['meal_plan'] = "Not Selected"
    elif meal_sel[0] not in ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']:
        return False, "Invalid Meal Plan selected"
    else:
        input_data['meal_plan'] = meal_sel[0]

    logger.info("Input validation successful")
    return True, ""

def main():
    # Header
    st.markdown('<h1 class="main-header">Hotel Reservation Cancellation Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter booking details to predict cancellation probability</p>', unsafe_allow_html=True)

    # Load model
    model = load_model()

    # Create two main columns with a gap: left for inputs, gap, right for button and results
    input_col, gap_col, result_col = st.columns([1.8, 0.3, 1])

    with input_col:
        st.markdown("### Booking Details")
        input_data = {}

        # First row: Lead time and price
        c1, gap, c2 = st.columns([1, 0.2, 1])
        with c1:
            input_data['lead_time'] = st.slider(
                'Lead Time (days)',
                min_value=0, max_value=500, value=7,
                help="Days between booking and arrival (0–500)"
            )
            st.caption("Allowed range: 0 — 500")

            input_data['avg_price_per_room'] = st.slider(
                'Average Price per Room (€)',
                min_value=20, max_value=500, value=75,
                help="Average price per night (20–500 €)"
            )
            st.caption("Allowed range: 20 — 500 €")
        with c2:
            input_data['no_of_special_requests'] = st.slider(
                'Number of Special Requests',
                min_value=0, max_value=10, value=0,
                help="Total special requests (0–10)"
            )
            st.caption("Allowed range: 0 — 10")

        st.markdown("### Guest Information")

        # Second row: Number of adults and children
        c3, gap2, c4 = st.columns([1, 0.2, 1])
        with c3:
            input_data['no_of_adults'] = st.slider(
                'Number of Adults',
                min_value=0, max_value=5, value=1,
                help="Number of adults in the booking (0–5)"
            )
            st.caption("Allowed range: 0 — 5")
        with c4:
            input_data['no_of_children'] = st.slider(
                'Number of Children',
                min_value=0, max_value=5, value=0,
                help="Number of children in the booking (0–5)"
            )
            st.caption("Allowed range: 0 — 5")

        st.markdown("### Stay Details")

        # Third row: Weekend and week nights
        c5, gap3, c6 = st.columns([1, 0.2, 1])
        with c5:
            input_data['no_of_weekend_nights'] = st.slider(
                'Number of Weekend Nights',
                min_value=0, max_value=7, value=1,
                help="Number of weekend nights in stay (0–7)"
            )
            st.caption("Allowed range: 0 — 7")
        with c6:
            input_data['no_of_week_nights'] = st.slider(
                'Number of Week Nights',
                min_value=0, max_value=30, value=3,
                help="Number of weekday nights in stay (0–30)"
            )
            st.caption("Allowed range: 0 — 30")

        st.markdown("### Preferences")

        # Fourth row: Meal plan and market segment
        c7, gap4, c8 = st.columns([1, 0.2, 1])
        with c7:
            input_data['meal_plan'] = st.multiselect(
                'Select Meal Plan (choose at most one)',
                options=['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
                default=[]
            )
            if len(input_data['meal_plan']) > 1:
                st.error("Please select only one Meal Plan or none (Not Selected).")
        with c8:
            input_data['market_segment'] = st.selectbox(
                'Select Market Segment',
                options=['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'],
                index=0,
                help="Choose the booking market segment (single selection)."
            )
            st.caption("Single selection required")

    # Right column for prediction button and results
    with result_col:
        st.markdown("### Prediction")
        st.markdown("")  # Add spacing
        
        if st.button('🔮 Predict Cancellation', use_container_width=True, key='predict_btn'):
            is_valid, error_message = validate_inputs(input_data)
            if not is_valid:
                st.error(f"⚠️ {error_message}")
            else:
                try:
                    prediction, probability = predict_cancellation(input_data, model)

                    # Display results
                    st.markdown("### Results")
                    cancellation_prob = probability[1]
                    
                    if cancellation_prob >= CANCELLATION_RISK_THRESHOLD:
                        st.markdown(
                            f"""<div class="prediction-box warning-box">
                                <h2>⚠️ High Risk</h2>
                                <h3>{cancellation_prob:.1%}</h3>
                            </div>""", unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""<div class="prediction-box success-box">
                                <h2>✅ Low Risk</h2>
                                <h3>{cancellation_prob:.1%}</h3>
                            </div>""", unsafe_allow_html=True
                        )

                    st.markdown("### Details")
                    st.metric("Keep Reservation", f"{probability[0]:.1%}")
                    st.metric("Cancellation", f"{probability[1]:.1%}")
                    st.caption(f"Risk threshold: {CANCELLATION_RISK_THRESHOLD:.1%}")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Sidebar with GitHub link
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("[🔗 GitHub Repository](https://github.com/rolaseba/hotel-reservation-prediction)")
        st.markdown("---")
        st.markdown("### 📄 License")
        st.caption("MIT License © 2024 Sebastián Rolando")
        st.caption("Hotel Reservation Cancellation Prediction System")


if __name__ == '__main__':
    main()