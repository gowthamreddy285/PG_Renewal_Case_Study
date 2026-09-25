# BUSINESS ANALYTICS INDIVIDUAL CASE STUDY

# The Renewal Threshold: Predicting PG Resident Renewal Intention Using Behavioral and Service-Experience Analytics

**Analytical Dataset:** Supplied 10,000-row synthetic survey dataset
**Report Basis:** `analysis.ipynb`

---

## 1. Problem Statement and Objectives

### 1.1 Business Problem and Strategic Context
The modern Paying Guest (PG) accommodation, co-living, and short-term residential leasing sector has become increasingly saturated and fiercely competitive. Operators face significant challenges in maximizing resident lifetime value (LTV) while mitigating the exorbitant costs associated with high churn rates. Customer acquisition costs (CAC)—including marketing, brokerage fees, and administrative onboarding overhead—severely erode profit margins if a resident vacates after a short tenure. 

Traditionally, the sector has relied heavily on capital-intensive infrastructure to attract and retain tenants, prioritizing physical amenities (e.g., high-speed Wi-Fi, air conditioning, premium furnishings) and tangible security features (e.g., biometric access, 24/7 guards). However, empirical operational feedback suggests a paradigm shift: while physical amenities drive initial acquisition, ongoing service experiences—specifically conflict resolution and perceived fairness—dictate long-term retention. This study explores the hypothesis that a resident's service journey (measured by complaint handling efficacy, total complaint volume, deposit transparency, and rent fairness perception) provides significantly more predictive information regarding their intention to renew than static amenity and security ratings.

### 1.2 Significance of the Research
Addressing this problem fundamentally alters how property management firms allocate capital. If service experiences are proven to be superior predictors of churn, operators can strategically pivot investments from high-cost, low-yield physical upgrades toward responsive customer relationship management (CRM) systems and rapid-deployment maintenance teams. Furthermore, establishing quantitative thresholds (e.g., the exact number of unresolved complaints before a resident's probability of renewal drops below 50%) enables proactive, data-driven intervention protocols.

### 1.3 Specific Objectives
This case study is guided by three primary objectives:
1. **Descriptive & Exploratory Profiling:** To systematically profile the resident demographic, analyze the distribution of the primary target variable (Renewal Intention), and identify preliminary correlations between independent operational variables and the target.
2. **Predictive Modeling & Feature Evaluation:** To rigorously compare the predictive efficacy of an "Amenity-Only" feature block against a "Service-Only" feature block, ultimately developing a "Combined" model utilizing Logistic Regression and Random Forest algorithms.
3. **Strategic Business Formulation:** To translate complex statistical outputs and feature importance rankings into actionable, low-risk, and high-impact operational recommendations for PG management.

---

## 2. Data Collection and Dataset Description

### 2.1 Data Collection Methodology
*Note: In alignment with the approved project proposal, the analysis is conducted on a supplied 10,000-row synthetic dataset designed to mirror a comprehensive, real-world tenant satisfaction survey.*

The data architecture simulates an exhaustive online questionnaire administered to current PG residents. In a practical deployment, this data would be harvested via a digital survey platform (e.g., Qualtrics or Google Forms) distributed to residents who have surpassed a minimum tenure threshold (e.g., 3 months), ensuring they possess sufficient longitudinal experience to accurately evaluate the facility's service infrastructure.

### 2.2 Dataset Attributes and Structural Integrity
The primary analytical file (`data/synthetic_survey_10000.csv`) comprises 10,000 records across 12 highly structured dimensions, encompassing demographic, financial, physical, and qualitative service metrics:
- **Demographic & Tenure:** `Occupation` (Working Professional, Student), `Stay Duration` (continuous months).
- **Financial Perception:** `Monthly Rent` (numeric value), `Rent Fairness` (categorical: Underpriced, Fair, Overpriced), `Deposit Refundability` (categorical: Straightforward, Not sure, Disputes common).
- **Physical Property Ratings (1-5 Likert Scale):** `Amenities Rating`, `Security Rating`.
- **Operational Service Experience:** `Complaint Count` (integer representing total issues raised), `Complaint Resolution` (categorical: Same day, Few days, Over a week, Never resolved, No complaints raised).
- **Target Variables:** `Overall Satisfaction` (1-5 scale) and the primary classification target, `Renewal Intention` (categorical: Yes, Maybe, No).
- **Qualitative Context:** `Biggest Reason` (free-text string).

### 2.3 Distribution and Analytical Scope
The demographic distribution skews heavily toward young urban earners, comprising 9,145 working professionals and 855 students. The primary target variable, `Renewal Intention`, exhibits the following distribution:
- **Yes:** 4,706 (47.06%)
- **Maybe:** 4,194 (41.94%)
- **No:** 1,100 (11.00%)

To facilitate rigorous binary predictive modeling, the ambiguous "Maybe" responses were intentionally excluded. This isolates unequivocal positive and negative intentions, yielding a highly decisive modeling subset of 5,806 records. Within this subset, a significant class imbalance is present (81.05% Yes, 18.95% No), necessitating specialized algorithmic weighting during the modeling phase.

---

## 3. Data Preparation and Exploratory Analysis

### 3.1 Advanced Cleaning and Preprocessing Pipeline
Robust data preprocessing was executed within the `analysis.ipynb` environment to ensure algorithmic stability:
1. **Missing Values and Duplicate Handling:** The raw matrix contained zero null values. While 69 exact duplicate rows were detected, they were retained. Given the synthetic nature of the dataset and the absence of unique cryptographic identifiers, these duplicates represent natural probabilistic response overlaps rather than erroneous duplicate entries.
2. **Mathematical Variable Encoding:** 
   - Ordinal categorical variables were mapped to preserve inherent hierarchy. `Rent Fairness` was encoded sequentially (Underpriced=0, Fair=1, Overpriced=2).
   - `Deposit Refundability` was structured from seamless (0) to highly frictional (2).
   - `Complaint Resolution` required nuanced encoding to differentiate between rapid resolution (0), delayed resolution (1-3), and the complete absence of complaints, which was isolated as a mathematically distinct model category (4) to prevent algorithmic confusion.
3. **Target Leakage Mitigation:** The `Biggest Reason` free-text field was strictly omitted from all predictive matrices to prevent target leakage. Similarly, `Overall Satisfaction` was isolated as a proxy variable; including it would artificially inflate model accuracy, as satisfaction is nearly perfectly collinear with renewal intention.

### 3.2 Exploratory Data Analysis (EDA) and Key Insights
Exploratory visualization yielded profound, actionable insights prior to the application of machine learning.

- **The Amenity Gradient:** A positive correlation exists between physical amenities and renewal. At a rating of 2, only 32.4% express a definite "Yes" for renewal. This scales to 54.9% at a rating of 5. While positive, the gradient is surprisingly shallow, suggesting amenities are necessary but not sufficient for retention.
- **The Service Experience Cliff:** The most statistically dominant pattern emerges in complaint volume. 
  - Residents with **0 complaints**: 64.3% "Yes" renewal rate.
  - Residents with **2 complaints**: Rate decays to 32.9%.
  - Residents with **4 complaints**: Rate collapses to 10.6%.
  *Interpretive Insight:* A distinct psychological "tolerance threshold" exists. Recurring operational failures fundamentally fracture resident trust, overriding any positive sentiment generated by high-quality physical amenities.
- **Financial Perception over Absolute Cost:** The absolute numeric value of rent is less predictive than the *perception* of fairness. Residents categorizing their rent as "Fair" exhibit a 49.5% renewal rate, which precipitously drops to 20.2% for those perceiving it as "Overpriced."

---

## 4. Analytics Method and Implementation

### 4.1 Justification of Analytical Methodologies
The study utilizes a tiered analytical approach, blending non-parametric inferential statistics with advanced supervised machine learning.
1. **Mann–Whitney U Tests & Chi-Square Analysis:** Deployed to establish foundational statistical significance. These tests confirm whether the distributions of independent variables materially differ between the "Yes" and "No" cohorts.
2. **Logistic Regression (Linear Baseline):** Selected for its profound interpretability. It allows management to understand the precise directional coefficients (e.g., the exact log-odds decay associated with an additional complaint).
3. **Random Forest Classifier (Non-Linear Ensemble):** Utilized to map complex, non-linear feature interactions (e.g., the compound effect of high rent coupled with slow resolution). The Random Forest also generates mathematically rigorous permutation feature importance rankings.

### 4.2 Implementation and Algorithmic Validation
Modeling was implemented via the `scikit-learn` ecosystem.
- **Cross-Validation Framework:** To guarantee out-of-sample reliability, a 5-fold Stratified Cross-Validation protocol, repeated 10 times (yielding 50 distinct held-out folds), was engineered.
- **Algorithmic Class Balancing:** To combat the 81% / 19% class imbalance, class weights were algorithmically balanced. Without this, a naive model would simply predict "Yes" universally, achieving 81% superficial accuracy while failing completely at identifying churners.
- **Performance Metrics:** Raw accuracy was explicitly discarded. Models were evaluated on **ROC-AUC** (discriminative capacity), **Balanced Accuracy** (average of sensitivity and specificity), and **Average Precision** (reliability of positive predictions).

---

## 5. Comparison with State-of-the-Art Methods

To ground this research in contemporary academic literature, the methodology and findings are benchmarked against three recent, high-impact studies in property management and hospitality.

| Published Study / Year | Dataset | Method Used | Evaluation Metric | Key Result | Comparison with This Work |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In, S. (2024)**<br>*Impact of commercial real estate service quality on tenants' intention to renew* | Commercial real-estate tenant surveys | Structural Equation Modeling (SEM) | Statistical Significance (P-Values) | Management reliability, adequacy, and tenant empathy positively dictate renewal. | **High Contextual Overlap.** While they rely on abstract constructs ("empathy"), our study operationalizes this via quantifiable metrics (resolution speed, complaint volume) using predictive machine learning. |
| **Marandi, Tasavori & Najmi (2024)**<br>*Hotel revisiting intentions based on big data* | ~100,000 TripAdvisor reviews globally | Natural Language Processing (NLP) | Text segmentation & clustering | Identified 71 features; staff interaction and accessibility were highly salient. | **Hospitality Context.** They focus on unstructured text mining. Conversely, our study utilizes a highly structured survey to test predefined operational hypotheses (Amenity vs. Service) via predictive Cross-Validation. |
| **Park, Y.-J., & Lee, Y.-D. (2025)**<br>*Housing location factors on lease renewal for single households* | Survey of single-person households in Busan | Mediation analysis | Significance/Mediation tests | Macro-location factors heavily influence renewal; mediated by satisfaction. | **Residential Context.** They analyze immutable geographic factors. Our study empowers managers by focusing exclusively on highly mutable, controllable operational variables. |

---

## 6. Results, Business Insights, and Recommendations

### 6.1 Comprehensive Model Performance
The predictive modeling phase conclusively validated the core research hypothesis. The repeated cross-validation yielded the following performance metrics:

1. **Feature Block Superiority (ROC-AUC):**
   - **Amenity & Security Only:** Logistic Regression: 0.660 | Random Forest: 0.654
   - **Service Experience Only:** Logistic Regression: 0.787 | Random Forest: 0.806
   - **Combined (Comprehensive):** Logistic Regression: 0.831 | **Random Forest: 0.842**

2. **Precision and Imbalance Resolution:**
   - The naive majority-class baseline yields a superficial accuracy of 0.811, but a mathematically useless balanced accuracy of 0.500.
   - The Combined Random Forest achieved a **Balanced Accuracy of 0.762**, demonstrating exceptional capability in isolating the minority class (churners). 
   - The **Average Precision metric reached 0.950**, indicating that when the model flags a resident as highly probable to renew, the prediction is overwhelmingly reliable.

3. **Permutation Feature Importance:**
   Rigorous permutation testing on the Random Forest revealed that **Complaint Count** is the apex predictor, causing a massive 0.223 deterioration in ROC-AUC when randomized. This was distantly followed by Amenities (0.048) and Rent Unfairness (0.037).

### 6.2 Strategic Business Insights
The empirical evidence necessitates a strategic paradigm shift: **Physical infrastructure functions as an acquisition tool, but operational service infrastructure acts as the retention engine.** 

The fact that the "Service Only" feature block outperformed the "Amenity Only" block by a staggering ~0.150 ROC-AUC points proves that tenant retention is won and lost in the operations backend. A premium PG facility suffering from chronic, unresolved maintenance issues will experience higher financial churn than a mid-tier facility boasting rapid, transparent issue resolution. 

### 6.3 Actionable Operational Recommendations
Based on the statistical findings, PG operators should immediately deploy the following strategic frameworks:

* **Recommendation 1: Deploy a "Threshold-Triggered" CRM Alert Protocol.** 
  Given the empirical collapse in renewal probability after a resident logs their second complaint, operators must abandon passive ticketing systems. Implement an automated CRM trigger: any resident filing a 3rd maintenance ticket must trigger an immediate escalation to senior management, accompanied by a mandated "courtesy check-in" call, bypassing standard queue times to salvage the relationship.
* **Recommendation 2: Reallocate Capital Expenditure (CapEx) to Operational Expenditure (OpEx).**
  Operators seeking to maximize ROI should freeze investments in marginal physical upgrades (e.g., cosmetic room improvements). Capital should be reallocated to fund dedicated, rapid-response maintenance personnel. The data unequivocally proves that speed of resolution yields a higher retention ROI than minor physical luxury.
* **Recommendation 3: Mandate Financial Transparency at Onboarding.**
  Because rent *perception* overrides absolute rent cost in predicting churn, operators must aggressively manage this perception. Implement a standardized "Value Breakdown Statement" during the lease signing phase, clearly visualizing the exact cost offset of included utilities, Wi-Fi, and maintenance. This psychological anchoring shifts the resident's classification from "Overpriced" to "Fair," statistically bolstering retention probability.

---

## 7. Conclusion and References

### 7.1 Executive Conclusion
This individual case study successfully interrogated the primary drivers of PG resident renewal intention utilizing a robust 10,000-record dataset. Through extensive exploratory data analysis and rigorous supervised machine learning—specifically utilizing Logistic Regression and Random Forest classifiers with Repeated Stratified Cross-Validation—the research proved that behavioral and operational service factors vastly outperform static physical amenities in predictive capacity.

The Combined Random Forest model achieved an elite ROC-AUC of 0.842, an Average Precision of 0.950, and successfully managed extreme class imbalance to reach a Balanced Accuracy of 0.762. The empirical data overwhelmingly indicates that complaint volume and rent fairness are the apex predictors of tenant churn. Consequently, PG operators are strongly advised to pivot their strategic focus from physical capital expenditure toward transparent, high-velocity customer service operations to secure long-term profitability and maximize resident lifetime value.

### 7.2 Academic References
1. In, S. (2024). A study on the impact of commercial real estate management service quality on tenants’ intention to renew their contracts. *Asia-Pacific Journal of Convergent Research Interchange, 10*(8), 199–210. https://doi.org/10.47116/apjcri.2024.08.16
2. Marandi, A. A., Tasavori, M., & Najmi, M. (2024). New insights into hotel customers’ revisiting intentions, based on big data. *International Journal of Contemporary Hospitality Management, 36*(1), 292–311. https://doi.org/10.1108/IJCHM-06-2022-0719
3. Park, Y.-J., & Lee, Y.-D. (2025). The effect of housing location factors on lease renewal intention among single-person households. *Korean Journal of Construction Engineering and Management, 26*(5), 53–65. https://doi.org/10.6106/KJCEM.2025.26.5.053
4. Dataset Source: Supplied project analytical file `data/synthetic_survey_10000.csv` provided via faculty assignment infrastructure.
