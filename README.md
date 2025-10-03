# **Aircraft Engine Failure Prediction with Machine Learning**

## 1. Problem Statement  
Aircraft engine maintenance is costly and critical for safety. Traditionally, maintenance is scheduled on fixed intervals or when sensors cross thresholds, which can lead to unnecessary downtime or unexpected failures.  
This project aims to build a data-driven model to predict the **Remaining Useful Life (RUL)** of jet engines using sensor and operational data. The goal is to forecast when an engine will likely fail, so maintenance can be scheduled **just in time**, reducing costs and avoiding catastrophic failures.

## 2. Data Source  
Data is from NASA’s C-MAPSS (Commercial Modular Aero Propulsion System Simulation) dataset. 
You can access the dataset here: [NASA C-MAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

Specifically, I used the **FD002** subset, which includes multiple engines, variable cycle lengths, 3 operational settings, and ~21 sensor measurements per cycle.

## 3. Machine Learning Model Used and Why  

The final model used in this project is **XGBoost (Extreme Gradient Boosting)** As a baseline, a model using RandomForestRegressor is also implemented.

### Why XGBoost?  
- **Handles Nonlinearities**: Engine degradation is nonlinear; XGBoost captures interactions between sensors effectively.  
- **Robustness to Noise**: Sensor data can be noisy, and tree ensembles don’t require heavy preprocessing.  
- **Interpretability**: Feature importance scores provide insights into which sensors and engineered features matter most.  

## 4. How This Project Solves It  

### 4.1 Workflow Overview  
1. **Exploratory Data Analysis (EDA)**  
   - Examined sensor statistics, distributions, correlations  
   - Dropped redundant sensors  
   - Visualized engine lifetimes  

2. **Feature Engineering**  
   - Rolling statistics (mean, std, slope)  
   - Normalized cycles (progress fraction)  
   - PCA for dimensionality reduction  
   - RUL capping & transformations to stabilize training  

3. **Model Training & Tuning**  
   - XGBoost with **RandomizedSearchCV** for hyperparameter optimization  
   - Custom weighting to reduce error in safe/medium ranges  
   - Compared against simpler baselines  

4. **Evaluation & Error Analysis**  
   - Metrics: RMSE, MAE, R²  
   - NASA scoring function (piecewise penalty)  
   - Error breakdown by RUL zones  
   - Bias analysis and residual plots  

5. **Interpretability & Visualization**  
   - Feature importance ranking  
   - Actual vs Predicted plots  
   - Error distributions and zone-wise breakdown  

## 5. References

- NASA. (n.d.). CMAPSS Jet Engine Simulated Data. NASA. https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
  
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008, October). Damage propagation modeling for aircraft engine run-to-failure simulation. In 2008 International Conference on Prognostics and Health Management (pp. 1–9). IEEE. https://doi.org/10.1109/PHM.2008.4711414
  
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Metrics for evaluating performance of prognostic techniques. 2008 International Conference on Prognostics and Health Management, 1–17. https://doi.org/10.1109/PHM.2008.4711436

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD ’16. https://doi.org/10.1145/2939672.2939785

## Connect
Author: Akarsh Doki

Major: CS + Business Administration

[LinkedIn](https://www.linkedin.com/in/akarsh-doki-600a35282/)

Email: doki.ak@northeastern.edu
