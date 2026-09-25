# The Renewal Threshold: Predicting PG Resident Renewal Intention Using Behavioral and Service-Experience Analytics

## Problem statement
This case explores whether PG residents’ complaint handling, complaint volume, deposit refund transparency and rent fairness contain more predictive information about stated renewal intention than amenities and security ratings. Overall satisfaction is kept out of the primary feature comparison.

## Objectives
1. Describe the supplied dataset and its three renewal-intention categories.
2. Compare amenity-only, service-only and combined blocks using Logistic Regression and Random Forest.
3. Translate evidence into cautious operational recommendations.

## Data source
The supplied `pg_synthetic_10000.csv` was accepted by the faculty and is used as the primary analytical dataset. It contains 10,000 synthetic survey-like records, not 10,000 real respondents. The original Google Forms export and approved proposal were not available in the project files, so source-to-synthetic fidelity and original cleaning decisions could not be independently audited. “Maybe” responses are excluded from binary Yes/No models, not recoded as No.

## Dataset
- `data/synthetic_survey_10000.csv`: supplied 10,000-row synthetic analytical file.
- Original raw and cleaned survey exports are not included because they were not supplied for this work.

## Data preparation
Readable variables are derived in memory; the supplied file remains unchanged. Yes/No rows form the definite-intention modeling subset; Maybe remains visible in descriptive summaries. “No complaints raised” is represented as its own model category, distinct from “never resolved.” “Biggest reason” is excluded to prevent explanation leakage. Satisfaction is evaluated separately as a proxy/leakage experiment. No unexpected-charges variable is present or invented.

## Analytics methods
Exploratory distributions and renewal rates; Mann–Whitney U tests with rank-biserial effect sizes; chi-square association with Cramér’s V; Logistic Regression and Random Forest; 5-fold stratified cross-validation repeated 10 times; majority-class baseline; permutation feature importance; satisfaction inclusion sensitivity experiment.

## Key results
On the definite Yes/No subset (n=5,806; 4,706 Yes, 1,100 No), repeated 5-fold/10-repeat CV produced mean ROC-AUC of 0.660/0.654 (amenity-only Logistic Regression/Random Forest), 0.787/0.806 (service-only), and 0.831/0.842 (combined). Complaint count had the largest permutation importance in the combined Random Forest (mean ROC-AUC decrease 0.223 on one held-out split). Satisfaction inclusion increased ROC-AUC to 0.911/0.913, supporting its separate treatment as a proxy/leakage sensitivity. The supplied file also contains 4,194 Maybe responses, excluded from binary modeling, and 69 exact duplicate rows. All metrics describe the synthetic file and must not be interpreted as performance on new PG residents.

## Business insights
Use the analysis to identify service areas worth monitoring, not to infer causal effects or automate resident decisions. Validate patterns prospectively using an ethically collected sample and observed renewal outcomes before operational deployment.

## Data limitations
Faculty acceptance establishes the supplied synthetic file as the approved analytical dataset for this case. The original Google Forms export, approved proposal and verifiable synthesis procedure were not available in the project files. Accordingly, this project includes only the supplied synthetic analytical CSV; it does not create substitute raw or cleaned survey files and cannot compare source-vs-synthetic distributions or independently verify upstream cleaning.

## References
- In, S. (2024). A study on the impact of commercial real estate management service quality on tenants’ intention to renew their contracts. *Asia-Pacific Journal of Convergent Research Interchange, 10*(8), 199–210. https://doi.org/10.47116/apjcri.2024.08.16
- Marandi, A. A., Tasavori, M., & Najmi, M. (2024; online 2023). New insights into hotel customers’ revisiting intentions, based on big data. *International Journal of Contemporary Hospitality Management, 36*(1), 292–311. https://doi.org/10.1108/IJCHM-06-2022-0719
- Park, Y.-J., & Lee, Y.-D. (2025). The effect of housing location factors on lease renewal intention among single-person households. *Korean Journal of Construction Engineering and Management, 26*(5), 53–65. https://doi.org/10.6106/KJCEM.2025.26.5.053
