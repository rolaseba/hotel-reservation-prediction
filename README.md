# Hotel Reservation Cancellation Prediction

**Repo:** [https://github.com/rolaseba/hotel-reservation-prediction](https://github.com/rolaseba/hotel-reservation-prediction)  
**Author:** Sebastian Rolando

## 🎯 Executive Summary

**Business Result:** Developed and validated an ensemble machine learning model that predicts booking cancellations with **87.5% accuracy** and **86% sensitivity**, demonstrating a scalable solution to address the $670K annual revenue loss from cancellations in mid-sized hotels. This portfolio project showcases a production-ready approach to proactive revenue protection through data-driven risk assessment.

## 📊 Dataset Context

**Origin and Source**
This analysis utilizes the publicly available Hotel Booking Demand dataset from two hotels in Portugal, providing real-world validation of the methodology:

- **H1**: A resort hotel in the Algarve region (southern Portugal, tourism-focused)
- **H2**: A city hotel in Lisbon (urban business/leisure mix)
- **Period**: Bookings scheduled between July 2015 - August 2017 (119,000+ observations)
- **Data Integrity**: Personally identifiable information removed for privacy; all data anonymized for research use

*Source: Hotel booking demand datasets [ResearchGate Publication](https://www.researchgate.net/publication/329286343_Hotel_booking_demand_datasets)*

**Note**: While this specific implementation uses historical Portuguese hotel data, the methodology, feature engineering, and model architecture are directly applicable to hotel operations globally, particularly addressing the universal challenge of cancellation-driven revenue loss.

## 💰 Hotel Cancellation Impact Use Case

In a mid-sized U.S. urban hotel with typical $10M annual revenue, cancellations and no-shows erode **6.7% of revenue ($670K loss yearly)**, with peaks up to **11% during high seasons** like summer due to unresold rooms and distorted forecasting [[1]](https://skift.com/2024/07/23/summer-travels-hidden-hurdle-the-impacts-of-cancellations-and-no-shows/) [[2]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5051585). 

**Channel Segmentation:** OTA channels (e.g., Booking.com) drive higher impacts, with cancellations accounting for **40% of booking revenue** in regions like APAC, compared to lower rates for direct bookings [[3]](https://www.d-edge.com/wp-content/uploads/2024/04/Hotel-Distribution-Report-2024-EN.pdf). 

**Guest Behavior:** Gen Z and Millennials contribute more (**64-66% seeking flexibility**), often rebooking at lower rates after price drops. A **$50 rate hike boosts cancellation risk by 16%**, turning potential profit into losses from discounts or voids [[2]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5051585).

**High-risk OTA:** An online travel agency that contributes significantly to cancellation-driven revenue loss, forecasting distortion, and profitability erosion due to flexible booking terms, guest price sensitivity, and volatile demand behavior.

## 🏆 Solution Performance & Validation

### Optimal Model: Voting Classifier with Under-Sampling
- **Overall Accuracy:** 87.5%
- **Sensitivity (Cancellation Detection):** 86%
- **Specificity (Booking Confidence):** 88%
- **F1-Score:** 0.83

**Validation Approach**: Trained and tested on real hotel booking data, demonstrating robust performance across both resort and city hotel contexts, with methodology directly transferable to other hotel markets.

## 📈 Key Business Insights

### Critical Cancellation Drivers Identified:
1. **Lead Time Effect:** Longer advance bookings show 3.2x higher cancellation risk
2. **Guest Loyalty Impact:** Repeat guests demonstrate 68% lower cancellation rates
3. **Service Engagement:** Special requests correlate with 45% reduced cancellations
4. **Seasonal Patterns:** High-risk months identified for targeted interventions
5. **Pricing Sensitivity:** $50 rate increases linked to 16% higher cancellation risk

## 🚀 Business Applications & Transferability

### Immediate Use Cases:
- **Revenue Protection:** Dynamic overbooking based on cancellation risk scores
- **Channel Optimization:** Targeted management of high-risk OTA bookings (40% cancellation revenue impact)
- **Guest Segmentation:** Proactive handling of Gen Z/Millennial segments (64-66% flexibility seekers)
- **Pricing Strategy:** Data-informed rate decisions to minimize 16% cancellation risk from price hikes

### Methodology Transferability:
- **Feature Engineering:** Universal booking attributes (lead time, guest history, channel source)
- **Model Architecture:** Adaptable to any hotel's specific data schema
- **Deployment Framework:** Scalable across property types and management systems

---

## 🔬 Technical Deep Dive

### Model Comparison Metrics

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | Sensitivity |
|-------|----------|-----------|--------|----------|-------------|-------------|
| **Initial Random Forest** | 89% | 0.84 | 0.72 | 0.78 | 95% | 72% |
| **Under-Sampled RF** | 87% | 0.80 | 0.86 | 0.83 | 88% | 86% |
| **Under-Sampled XGB** | 85% | 0.78 | 0.84 | 0.81 | 86% | 84% |
| **Under-Sampled CatBoost** | 86% | 0.79 | 0.85 | 0.82 | 87% | 85% |
| **Voting Classifier (Under)** 🏆 | **87.5%** | **0.81** | **0.86** | **0.83** | **88%** | **86%** |
| **Over-Sampled RF** | 88% | 0.82 | 0.84 | 0.83 | 90% | 84% |
| **Voting Classifier (Over)** | 88.2% | 0.83 | 0.84 | 0.83 | 90% | 84% |

### Model Selection Rationale

**Business-Driven Technical Decision:**

The **Voting Classifier with Under-Sampling** was selected based on comprehensive cost-benefit analysis:

- **Initial Random Forest (89% accuracy, 72% sensitivity):** Achieved high accuracy by optimizing for majority class, but missed 28% of cancellations - creating significant revenue leakage
- **Voting Classifier (87.5% accuracy, 86% sensitivity):** Sacrificed 1.5% overall accuracy to gain 14% improvement in cancellation detection

**Technical Justification:**
- **Higher Business Value:** False negatives (missed cancellations) have greater financial impact than false positives
- **Ensemble Robustness:** Voting classifier reduces variance and leverages complementary model strengths
- **Optimal Trade-off:** Balanced specificity (88%) and sensitivity (86%) for operational practicality

### Technical Stack

```python
# Core Libraries
pandas, numpy, matplotlib, seaborn

# Machine Learning
scikit-learn, imblearn, xgboost, catboost

# Specialized Techniques
PCA (Feature Engineering)
RobustScaler (Feature Scaling)
SVMSMOTE (Over-sampling)
VotingClassifier (Ensemble Learning)
```

### Feature Engineering

**Advanced Transformations:**
- PCA for correlated features (`no_of_people`, `no_of_week_days`)
- Business-informed categorical groupings
- Historical cancellation rate calculation

**Preprocessing Pipeline:**
- RobustScaler for numerical features
- Strategic one-hot encoding
- Correlation-based feature selection

### Ensemble Architecture

**Voting Classifier Configuration:**
- **Random Forest** (weight: 2.5) - Primary robust ensemble
- **CatBoost** (weight: 1.5) - Categorical feature specialist  
- **XGBoost** (weight: 1.0) - Gradient boosting performance
- **Soft Voting** - Probability-based predictions

## 🚀 Deployment Framework

**Option 1: Batch API** - Daily prediction service generating cancellation risk scores for upcoming arrivals

**Option 2: Real-time API** - REST endpoint integrated with booking system for instant risk assessment on new reservations

**Option 3: Automated Reporting** - Scheduled reports delivered to revenue teams with high-risk booking alerts and insights

**Monitoring & Maintenance** - Weekly performance tracking with monthly model retraining and automated alerting

---

**Portfolio Value:** This project demonstrates a complete, production-ready machine learning pipeline validated on real hotel data. The methodology, feature engineering approach, and model architecture are directly transferable to hotel operations globally, providing a proven framework for addressing the universal $670K revenue loss challenge from booking cancellations.

*Complete implementation code, experimental results, and methodology documentation available in the project repository.*