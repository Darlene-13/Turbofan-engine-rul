
## Predictive Maintenance and remaining useful life (RUL) estimation project for industrial Turbo Engines using Machine Learning


### Problem Statement
> Industrial turbofan engines are critical assets in aerospace, power generation, and heavy industrial applications. These engines operate under extreme mechanical and thermal conditions, leading to gradual degradation of components such as turbines, compressors, and bearings. Traditionally, maintenance of turbofan engines is performed either reactively after failure or preventively based on fixed schedules. While reactive maintenance increases the risk of catastrophic failure and unplanned downtime, scheduled preventive maintenance often results in unnecessary servicing, increased operational costs, and inefficient asset utilization.
>
> Modern turbofan engines are equipped with numerous sensors that continuously monitor operational parameters such as temperature, pressure, and rotational speed. Despite the availability of this large volume of sensor data, many maintenance strategies do not fully exploit it to predict engine health and remaining service life. Consequently, failures may still occur without sufficient early warning, and maintenance decisions are often made without accurate knowledge of the actual degradation state of the engine.
>
>There is therefore a need for an intelligent predictive maintenance system capable of analyzing historical sensor data to estimate the Remaining Useful Life (RUL) of turbofan engines. By leveraging machine learning techniques, such a system can identify degradation patterns, forecast impending failures, and support data-driven maintenance decisions. This project seeks to address this need by developing and evaluating machine learning models for accurate RUL estimation using publicly available industrial turbofan engine datasets.

### Aim of the Project
The main aim of this project is to design and implement a machine learning–based predictive maintenance system that estimates the Remaining Useful Life of industrial turbofan engines using sensor data.

#### Objectives of the Project
The specific and achievable objectives of this project are to:
1.	Study the operational behavior and degradation mechanisms of turbofan engines in order to understand how sensor measurements reflect engine health over time.
2.	Acquire and preprocess a public industrial turbofan engine dataset, including data cleaning, normalization, and transformation into a form suitable for machine learning analysis.
3.	Perform exploratory data analysis (EDA) to identify trends, correlations, and degradation patterns in sensor data associated with engine wear and failure.
4.	Develop meaningful features from raw sensor signals that capture the temporal and physical characteristics of engine degradation.
5.	Implement and train machine learning models such as Linear Regression, Random Forest, Gradient Boosting, and Long Short-Term Memory (LSTM) networks for Remaining Useful Life prediction.
6.	Evaluate and compare model performance using appropriate regression metrics such as Root Mean Square Error (RMSE) and Mean Absolute Error (MAE).
7.	Interpret model outputs in an engineering context, identifying key sensors and operating conditions that significantly influence engine degradation and failure.
8.	Design a decision-support framework that translates RUL predictions into maintenance actions, such as continued operation, scheduled maintenance, or shutdown.
9.	Assess the limitations and assumptions of the proposed system, including data constraints, model generalization, and practical deployment challenges.

#### Expected Outcomes
- A validated machine learning model capable of estimating the Remaining Useful Life of turbofan engines.
- Improved understanding of sensor-based degradation behavior in industrial engines.
- A predictive maintenance framework that can reduce unplanned downtime and maintenance costs.

Dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

### PROJECT SET UP
1. Download the nasa_cmapps dataset from: [Nasa_Cmapps_Data](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
2. Unzip the file and save the unziped file as nasa_cmaps_txt inside the raw/data folder
3. Convert the .txt files to csv.

### EXPLANATION OF THE DATASET

#### Dataset description
Engine degradation simulation was carried out using C-MAPSS. Four different were sets simulated under different combinations of operational conditions and fault modes. Records several sensor channels to characterize fault evolution. The data set was provided by the Prognostics CoE at NASA Ames.

                                                                                                                                                                                                                                                           +