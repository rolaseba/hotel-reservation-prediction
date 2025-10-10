# Hotel Reservation Cancellation Prediction


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)



**Repo:** [https://github.com/rolaseba/hotel-reservation-prediction](https://github.com/rolaseba/hotel-reservation-prediction)  
**Author:** Sebastian Rolando

> [!NOTE]
> This repository is part of a professional Data Science portfolio demonstrating applied machine learning for real-world business problems in the hospitality industry.


## Executive Summary

**Business Result:** Developed and validated an ensemble machine learning model that predicts booking cancellations with **87.5% accuracy** and **86% sensitivity**, demonstrating a scalable solution to address the $670K annual revenue loss from cancellations in mid-sized hotels. This portfolio project showcases a production-ready approach to proactive revenue protection through data-driven risk assessment.

## Dataset Context

**Origin and Source**
This analysis utilizes the publicly available Hotel Booking Demand dataset from two hotels in Portugal, providing real-world validation of the methodology:

- **H1**: A resort hotel in the Algarve region (southern Portugal, tourism-focused)
- **H2**: A city hotel in Lisbon (urban business/leisure mix)
- **Period**: Bookings scheduled between July 2015 - August 2017 (119,000+ observations)
- **Data Integrity**: Personally identifiable information removed for privacy; all data anonymized for research use

*Source: Hotel booking demand datasets [ResearchGate Publication](https://www.researchgate.net/publication/329286343_Hotel_booking_demand_datasets)*

**Note**: While this specific implementation uses historical Portuguese hotel data, the methodology, feature engineering, and model architecture are directly applicable to hotel operations globally, particularly addressing the universal challenge of cancellation-driven revenue loss.

> [!TIP]
> You can adapt this dataset structure for other domains, such as airline reservations or event bookings, using the same feature engineering techniques.


## Business Context: The Impact of Cancellations

In a typical urban hotel generating **$10M in annual revenue**, cancellations and no-shows result in **6.7% revenue loss ($670K annually)**, with peaks of **11% during high-demand seasons**.

### Key Factors
- **Distribution Channels:** OTAs (e.g., Booking.com) contribute up to **40% of cancellation-driven losses**.  
- **Guest Segments:** Gen Z and Millennials exhibit **64–66% flexibility preference**, often rebooking at lower prices.  
- **Pricing Sensitivity:** A **$50 rate increase** raises cancellation probability by **16%**.

> [!NOTE]
> On average, cancellations and no-shows can erode more than **6% of annual revenue** in mid-sized hotels — a margin often greater than yearly operational cost reductions.

**References:**  
[Skift – Impacts of Cancellations and No-Shows](https://skift.com/2024/07/23/summer-travels-hidden-hurdle-the-impacts-of-cancellations-and-no-shows/)  
[SSRN – Price Sensitivity and Cancellations](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5051585)  
[D-Edge – Hotel Distribution Report 2024](https://www.d-edge.com/wp-content/uploads/2024/04/Hotel-Distribution-Report-2024-EN.pdf)


## Solution Performance & Validation

### Optimal Model: Voting Classifier with Under-Sampling
- **Overall Accuracy:** 87.5%
- **Sensitivity (Cancellation Detection):** 86%
- **Specificity (Booking Confidence):** 88%
- **F1-Score:** 0.83

**Validation Approach**: Trained and tested on real hotel booking data, demonstrating robust performance across both resort and city hotel contexts, with methodology directly transferable to other hotel markets.

## Key Business Insights

### Critical Cancellation Drivers Identified:
1. **Lead Time Effect:** Longer advance bookings show 3.2x higher cancellation risk
2. **Guest Loyalty Impact:** Repeat guests demonstrate 68% lower cancellation rates
3. **Service Engagement:** Special requests correlate with 45% reduced cancellations
4. **Seasonal Patterns:** High-risk months identified for targeted interventions
5. **Pricing Sensitivity:** $50 rate increases linked to 16% higher cancellation risk

## Business Applications & Transferability

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

## Technical Deep Dive

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

The **Voting Classifier with Under-Sampling** was selected for its optimal trade-off between accuracy and sensitivity.  
It minimizes missed cancellations (false negatives) that represent the highest financial risk.

> [!TIP]
> The chosen ensemble model prioritizes **sensitivity** (detecting cancellations) over **overall accuracy**, reflecting a business-driven decision: missing a cancellation is costlier than flagging a false one.

**Advantages:**
- Improved detection of high-risk bookings  
- Ensemble robustness from multiple model perspectives  
- Balanced performance (specificity 88%, sensitivity 86%)

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

## Deployment Framework Opportunities

**Option 1: Batch API** - Daily prediction service generating cancellation risk scores for upcoming arrivals

**Option 2: Real-time API** - REST endpoint integrated with booking system for instant risk assessment on new reservations

**Option 3: Automated Reporting** - Scheduled reports delivered to revenue teams with high-risk booking alerts and insights

**Monitoring & Maintenance** - Weekly performance tracking with monthly model retraining and automated alerting

---

**Portfolio Value:** This project demonstrates a complete, production-ready machine learning pipeline validated on real hotel data. The methodology, feature engineering approach, and model architecture are directly transferable to hotel operations globally, providing a proven framework for addressing the universal $670K revenue loss challenge from booking cancellations.

*Complete implementation code, experimental results, and methodology documentation available in the project repository.*

## License

This project is licensed under the [MIT License](./LICENSE) © 2024 Sebastián Rolando.  
You are free to use, modify, and distribute this software, provided that proper credit is given.
