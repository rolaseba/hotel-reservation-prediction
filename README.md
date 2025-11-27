# Hotel Reservation Cancellation Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Repository:** [https://github.com/rolaseba/hotel-reservation-prediction](https://github.com/rolaseba/hotel-reservation-prediction)  
**Author:** Sebastián Rolando | **License:** MIT © 2024

> [!NOTE]
> This is a professional Data Science portfolio project demonstrating end-to-end machine learning for real-world business problems in the hospitality industry.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Solution Overview](#solution-overview)
5. [Technical Approach](#technical-approach)
6. [Model Performance](#model-performance)
7. [Key Insights](#key-insights)
8. [Deployment](#deployment)
9. [Installation & Usage](#installation--usage)
10. [Project Structure](#project-structure)
11. [References](#references)

---

## Executive Summary

**Business Impact:** Developed and deployed an ensemble machine learning model predicting booking cancellations with **88.2% accuracy** and **84% sensitivity**, directly addressing the **$670K annual revenue loss** from cancellations in mid-sized hotels.

**Key Achievement:** Balanced precision (83%) and sensitivity (84%) to optimize retention campaign efficiency while minimizing false alarms that damage customer relationships.

**Solution:** Production-ready interactive Streamlit web application enabling real-time cancellation risk assessment for individual bookings.

---

## Business Problem

### The Challenge

In a typical urban hotel generating **$10M in annual revenue**, cancellations and no-shows result in **6.7% revenue loss ($670K annually)**, with peaks of **11% during high-demand seasons**.

### Key Business Drivers

| Factor | Impact | Insight |
|--------|--------|---------|
| **Distribution Channels** | 40% of losses | OTAs (e.g., Booking.com) drive highest cancellations |
| **Guest Segments** | 64-66% flexibility | Gen Z & Millennials rebook at lower prices |
| **Pricing Sensitivity** | +16% risk | Each $50 rate increase raises cancellation probability |

**References:**  
- [Skift – Cancellations & No-Shows Impact](https://skift.com/2024/07/23/summer-travels-hidden-hurdle-the-impacts-of-cancellations-and-no-shows/)
- [D-Edge – Hotel Distribution Report 2024](https://www.d-edge.com/wp-content/uploads/2024/04/Hotel-Distribution-Report-2024-EN.pdf)

---

## Dataset

### Source & Context

**Hotel Booking Demand Dataset** - Two Portuguese hotels (July 2015 - August 2017)

- **H1**: Resort hotel in the Algarve (tourism-focused)
- **H2**: City hotel in Lisbon (business/leisure mix)
- **Records**: 119,000+ booking observations
- **Privacy**: Fully anonymized, PII removed

*Source: [ResearchGate Publication](https://www.researchgate.net/publication/329286343_Hotel_booking_demand_datasets)*

### Class Distribution

- **Non-Canceled:** 67.2% (24,390 bookings)
- **Canceled:** 32.8% (11,885 bookings)
- **Challenge:** Significant class imbalance requiring SVMSMOTE balancing

### Key Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `lead_time` | Numeric | 0-500 days | Days between booking and arrival ⭐ Strong predictor |
| `avg_price_per_room` | Numeric | €20-500 | Average nightly rate |
| `no_of_special_requests` | Numeric | 0-10 | Special requests made |
| `no_of_adults` | Numeric | 0-5 | Adults in booking |
| `no_of_children` | Numeric | 0-5 | Children in booking |
| `no_of_weekend_nights` | Numeric | 0-7 | Weekend nights in stay |
| `no_of_week_nights` | Numeric | 0-30 | Weekday nights in stay |
| `type_of_meal_plan` | Categorical | 4 classes | Meal plan selected |
| `market_segment_type` | Categorical | 5 classes | Booking channel |
| `booking_status` | Binary | 0 or 1 | **TARGET:** 0=Not Canceled, 1=Canceled |

---

## Solution Overview

### 🏆 Optimal Model: Voting Classifier with Over-Sampling (SVM-SMOTE)

| Metric | Score | Business Impact |
|--------|-------|-----------------|
| **Accuracy** | 88.2% | Correct predictions overall |
| **Sensitivity (Recall)** | 84% | Detects 84% of actual cancellations |
| **Precision** | 83% | When flagged, 83% truly cancel |
| **Specificity** | 89.8% | Identifies 89.8% of loyal customers |
| **F1-Score** | 0.83 | Balanced performance |
| **False Positive Rate** | 10.2% | Only 10.2% loyal customers falsely flagged |

### Why This Model?

**Cost-Effectiveness**
- 83% precision = efficient retention budget allocation
- 10.2% FPR minimizes unnecessary retention offers
- Better ROI on campaigns

**Customer Experience**
- Low false alarm rate protects goodwill
- Avoids frustrating loyal customers
- Maintains brand reputation

**Balanced Performance**
- 84% sensitivity catches real cancellations
- 89.8% specificity identifies loyal guests
- Strategic advantage over short-term catch rates

---

Claro, aquí tienes la sección **Technical Approach** reescrita y optimizada.

He integrado los puntos de mejora que discutimos: explicamos el *porqué* de las decisiones técnicas (como el PCA dirigido y el Scaler robusto) para demostrar conocimiento del dominio, y he pulido la estructura para que se vea más profesional.

Puedes copiar y reemplazar toda la sección existente en tu `README.md` con esto:

-----

## Technical Approach

### Feature Engineering Strategy

**Targeted PCA (Dimensionality Reduction)**
Instead of applying PCA across the entire dataset, we targeted specific multicollinear subgroups to reduce noise while preserving interpretability:

  - **Guest Composition:** Combined `no_of_adults` + `no_of_children` → `no_of_people` (1 component).
  - **Stay Duration:** Combined `no_of_weekend_nights` + `no_of_week_nights` → `no_of_week_days` (1 component).
    *Impact:* Successfully reduced multicollinearity while preserving the total variance of these related features.

**Robust Scaling**
Selected `RobustScaler` over `StandardScaler` for numerical features.
*Rationale:* Hospitality data often contains extreme outliers in variables like `lead_time` and `avg_price_per_room`. RobustScaler uses statistics that are robust to outliers (scaling based on the Interquartile Range), preventing these extreme values from biasing the model.

**Feature Selection & Encoding**

  - **High-Value Features:** Retained strong predictors such as `lead_time`, `avg_price_per_room`, and `no_of_special_requests`.
  - **Noise Reduction:** Removed features with low predictive power (e.g., `arrival_date`, `required_car_parking_space`) based on feature importance analysis.
  - **Business Encoding:** Transformed seasonal variables (`arrival_month`) into risk-based categories based on historical cancellation rates.

### Preprocessing Pipeline

All preprocessing steps are encapsulated in a `ColumnTransformer` pipeline to ensure reproducibility and prevent data leakage during production inference:

```python
Pipeline Structure:
├── Numerical Features ...... RobustScaler
├── PCA Groups .............. RobustScaler + PCA (n_components=1)
└── Categorical Features .... OneHotEncoder (handle_unknown='ignore')
```

### Ensemble Architecture

We deployed a **Voting Classifier (Soft Voting)** to leverage the strengths of multiple algorithms:

```
                  ┌── Random Forest (Weight: 2.5) ──┐
Input Data ─────▶ │   CatBoost      (Weight: 1.5)   │ ──▶ Soft Voting ──▶ Final Prediction
                  └── XGBoost       (Weight: 1.0) ──┘
```

**Model Rationale:**

  - **Stability:** High weight on Random Forest (2.5) ensures robustness against noise and overfitting.
  - **Nuance:** CatBoost and XGBoost capture complex non-linear relationships and handle categorical variations effectively.
  - **Soft Voting:** Uses predicted probabilities rather than hard labels, allowing for more granular risk assessment and threshold adjustment.
---

## Model Performance

### Model Comparison: Under-Sampling vs. Over-Sampling (SVMSMOTE)

#### Under-Sampling Results

| Model | Accuracy | Precision | Recall | F1-Score | Specificity |
|-------|----------|-----------|--------|----------|-------------|
| Random Forest | 87% | 0.80 | 0.86 | 0.83 | 88% |
| XGBoost | 85% | 0.78 | 0.84 | 0.81 | 86% |
| CatBoost | 86% | 0.79 | 0.85 | 0.82 | 87% |
| **Voting Classifier** | **87.5%** | **0.81** | **0.86** | **0.83** | **87.3%** |

**Pros:** Highest recall (catches more cancellations)  
**Cons:** 12.7% false positive rate (loyalty impact)

#### Over-Sampling (SVMSMOTE) Results ✅ Selected

| Model | Accuracy | Precision | Recall | F1-Score | Specificity |
|-------|----------|-----------|--------|----------|-------------|
| Random Forest | 88% | 0.82 | 0.84 | 0.83 | 90% |
| XGBoost | 87% | 0.80 | 0.83 | 0.82 | 89% |
| CatBoost | 88% | 0.82 | 0.84 | 0.83 | 90% |
| **Voting Classifier** | **88.2%** | **0.83** | **0.84** | **0.83** | **89.8%** |

**Pros:** Better precision, lower FPR (10.2%), balanced metrics  
**Cons:** Slightly longer training time

### Why Over-Sampling Was Selected

1. **Better precision** → Efficient budget allocation
2. **Lower FPR** → Better customer relationships
3. **Long-term value** > short-term recall gain

---

## Key Insights

### Critical Cancellation Drivers (Ranked by Importance)

1. **Lead Time** ⏰ (r=0.44)
   - 3.2x higher cancellation risk for longer advance bookings
   - Early bookers show different patterns than last-minute reservations

2. **Special Requests** 📝 (r=-0.25)
   - 45% reduced cancellations with special requests
   - More requests = higher booking commitment

3. **Average Price** 💰
   - $50 rate increases → 16% higher cancellation risk
   - Dynamic pricing strategy needed

4. **Seasonal Patterns** 📅
   - High-risk months: March-July, September
   - Enable targeted interventions

5. **Guest Type** 👥
   - Repeat guests: 1.7% cancellation rate
   - New guests: 33.6% cancellation rate
   - Loyalty reduces cancellation risk 68%

### Feature Importance (Top 5)

```
lead_time:                15.2%  ████████████
avg_price_per_room:       12.8%  ███████████
no_of_special_requests:   11.5%  ██████████
market_segment_type:       9.7%  █████████
type_of_meal_plan:         8.3%  ████████
```

---

## Deployment

### 🚀 Interactive Web Application (Streamlit)

**Features:**
- **Real-time Predictions**: Input booking details → instant risk assessment
- **Configurable Threshold**: Default 60% (easily customizable)
- **Professional UI**: Color-coded risk indicators, intuitive interface
- **Live Probabilities**: Detailed breakdown of cancellation likelihood

**Running the App:**
```bash
streamlit run app.py
# Open browser to http://localhost:8501
```

**Configuration:**
```python
# Adjust in app.py
CANCELLATION_RISK_THRESHOLD = 0.60  # Change risk threshold
```

### Deployment Options

| Option | Use Case | Timeline |
|--------|----------|----------|
| **Batch API** | Daily risk reports | Overnight processing |
| **Real-time API** | Booking system integration | On-the-fly decisions |
| **Automated Reports** | Revenue team alerts | Daily/weekly briefings |

### Monitoring & Maintenance

- Weekly performance tracking
- Monthly model retraining with new data
- Automated alerts for performance degradation

---

## Model Evaluation Metrics

### Business-Critical Metrics (Primary Focus)

The following **4 metrics** directly impact business outcomes:

| Metric | Formula | Business Purpose |
|--------|---------|-----------------|
| **Recall** | TP / (TP + FN) | Detect actual cancellations |
| **Precision** | TP / (TP + FP) | Optimize retention budget |
| **Specificity** | TN / (TN + FP) | Identify loyal customers |
| **False Positive Rate** | FP / (FP + TN) | Protect customer experience |

### Business Implications

**Revenue Protection (Recall)**
- Missing 1 cancellation costs ~$300
- Need to catch as many cancellations as possible

**Budget Efficiency (Precision)**
- Each retention offer costs ~$5
- Only flag high-confidence cases
- Avoid wasting budget on sure-keepers

**Customer Relationships (Specificity & FPR)**
- False alarms frustrate loyal customers
- 10.2% FPR maintains goodwill
- Protects brand reputation

### 📉 Business Impact Simulation (Per 1,000 Bookings)

Based on the test set distribution (approx. 33% cancellation rate), here is the expected outcome for every 1,000 predictions:

| Customer Segment | Prediction | Count | Business Outcome |
|------------------|------------|-------|------------------|
| **Loyal Guests** (670 total) | Correct (TN) | **602** | ✅ Revenue Secured. No friction. |
| | **False Alarm (FP)** | **68** | ⚠️ Minor retention cost (10.2% error rate). |
| **Potential Cancellations** (330 total) | **Correct (TP)** | **277** | 💰 **Revenue Protected.** (84% detected). |
| | Missed (FN) | **53** | ❌ Revenue Lost (Opportunity cost). |


## Installation & Usage

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Quick Start

**1. Load and Use the Model:**
```python
import joblib
import pandas as pd

# Load pre-trained model
model = joblib.load('models/voting_classifier_pipeline_model.pkl')

# Prepare input
input_data = pd.DataFrame({
    'no_of_adults': [2],
    'no_of_children': [0],
    'no_of_weekend_nights': [1],
    'no_of_week_nights': [3],
    'lead_time': [100],
    'avg_price_per_room': [75],
    'no_of_special_requests': [1],
    'type_of_meal_plan': ['Meal Plan 1'],
    'market_segment_type': ['Online']
})

# Predict
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

print(f"Cancellation Risk: {probability[0][1]:.1%}")
```

**2. Run Interactive Web App:**
```bash
streamlit run app.py
# Open http://localhost:8501
```
![Alt text](reports/figures/app.png)

---

## Project Structure

```
hotel-reservation-prediction/
├── app.py                              # Streamlit web application
├── notebook.py                         # Complete analysis (Python format)
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── datasets/
│   └── Hotel Reservations.csv         # Original dataset
│
└── models/
    ├── voting_classifier_pipeline_model.pkl  # Trained model
    └── feature_names.pkl              # Feature reference
```

---

## Technical Stack

```python
# Core Libraries
pandas, numpy, matplotlib, seaborn, joblib

# Machine Learning
scikit-learn, imblearn, xgboost, catboost

# Deployment
streamlit

# Techniques
├── PCA (Dimensionality Reduction)
├── RobustScaler (Scaling)
├── SVMSMOTE (Oversampling)
└── VotingClassifier (Ensemble)
```

---

## Business Applications & Transferability

### Immediate Use Cases

✅ **Revenue Protection**
- Dynamic overbooking by cancellation risk
- Inventory allocation optimization

✅ **Channel Optimization**
- Target high-risk OTA bookings
- Direct booking incentives

✅ **Guest Segmentation**
- Personalized retention strategies
- Gen Z/Millennial targeting

✅ **Pricing Strategy**
- Dynamic pricing based on cancellation risk
- Rate optimization

### Methodology Transferability

- ✅ Universal booking attributes (lead time, history, channel)
- ✅ Adaptable to any hotel's data schema
- ✅ Scalable across property types (chain, independent, resorts)
- ✅ Applicable to other domains (airlines, events, rentals)

---

## References & Resources

### Dataset
- [Hotel Booking Demand - ResearchGate](https://www.researchgate.net/publication/329286343_Hotel_booking_demand_datasets)

### Technical Documentation
- [SVMSMOTE - Imbalanced Learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SVMSMOTE.html)
- [Voting Classifier - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Business References
- [Skift - Cancellations Report](https://skift.com/2024/07/23/summer-travels-hidden-hurdle-the-impacts-of-cancellations-and-no-shows/)
- [D-Edge - Distribution Report 2024](https://www.d-edge.com/wp-content/uploads/2024/04/Hotel-Distribution-Report-2024-EN.pdf)

---

## License

**MIT License © 2024 Sebastián Rolando**

You are free to use, modify, and distribute this software under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Questions & Support

**Have questions?** Open an issue on the [GitHub Repository](https://github.com/rolaseba/hotel-reservation-prediction)

**Last Updated:** 2024 | Sebastián Rolando | Data Science Portfolio
