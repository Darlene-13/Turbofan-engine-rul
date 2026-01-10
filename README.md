
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

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. 

The objective is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

**Overview of the Train dataset and test dataset column names** 

| Column | Name           | What                                        | Why                                     | Range                     | Predictive / Interpretation                  |
|--------|----------------|--------------------------------------------|----------------------------------------|---------------------------|---------------------------------------------|
| 1      | engine_id      | Unique identifier for each engine (1-100)  | Track which engine the reading belongs | 1-100                     | -                                           |
| 2      | cycle          | Operational cycle number (time step)       | Track progression through engine life  | 1-192 (varies by engine)  | -                                           |
| 3      | op_setting_1   | Operational Setting 1 (throttle resolver) | Affects engine performance             | -8 to 0                   | Engine load/speed condition                 |
| 4      | op_setting_2   | Operational Setting 2 (mach number)       | Flight altitude/speed effect           | -0.009 to 0.84            | Aerodynamic condition                        |
| 5      | op_setting_3   | Operational Setting 3 (altitude)          | Environmental effect on engine         | 0 to 42000 feet           | Flight altitude                              |
| 6      | T2             | Temperature Sensor 2 (fan inlet)           | Early indicator of engine health       | ~500-600 °R               | YES – rises before failure                   |
| 7      | T24            | Temperature Sensor 24 (LPC outlet)         | Compressor performance indicator        | ~600-700 °R               | YES – correlates with wear                   |
| 8      | T30            | Temperature Sensor 30 (HPC outlet)         | High-pressure compressor health         | ~1300-1700 °R             | YES – critical indicator                     |
| 9      | T50            | Temperature Sensor 50 (LPT outlet)         | Turbine performance                     | ~1200-1600 °R             | YES – degrades before failure                |
| 10     | P2             | Pressure Sensor 2 (fan inlet)              | Atmospheric/inlet condition            | ~0-5 psi                  | NO                                           |
| 11     | P15            | Pressure Sensor 15 (bypass duct)           | Bypass duct condition                   | ~0-50 psi                 | MODERATE                                     |
| 12     | P30            | Pressure Sensor 30 (HPC outlet)            | High-pressure compressor pressure       | ~200-600 psi              | YES – changes with degradation               |
| 13     | Nf             | Physical Fan Speed (RPM)                   | Fan operation speed                      | ~1500-3000 RPM            | YES – reduces as engine degrades            |
| 14     | Nc             | Physical Core Speed (RPM)                  | Core engine speed                        | ~2000-10000 RPM           | YES – highly correlates with failure        |
| 15     | epr            | Engine Pressure Ratio                        | Overall engine performance metric       | ~1.0-2.5                  | YES – key degradation indicator              |
| 16     | Ps30           | Static pressure at HPC outlet              | Compressor discharge static pressure    | ~0-500 psi                | YES – changes with wear                      |
| 17     | phi            | Fuel Flow Ratio                             | Engine load percentage                   | ~0-100%                   | NO                                           |
| 18     | NRf            | Normalized Fan Speed                        | Fan speed corrected for conditions      | ~0.5-1.0                  | YES – normalized version of Nf               |
| 19     | NRc            | Normalized Core Speed                        | Core speed corrected for conditions     | ~0.5-1.0                  | YES – normalized version of Nc               |
| 20     | BPR            | Bypass Pressure Ratio                        | Bypass duct pressure ratio               | ~5-15                     | MODERATE                                     |
| 21     | farB           | Fuel Air Ratio (Burner)                      | Combustor fuel-air ratio                 | ~0.015-0.030              | YES – changes with degradation               |
| 22     | htBleed        | Bleed Enthalpy                               | High-pressure compressor bleed valve    | ~300-400 BTU/lbm          | MODERATE                                     |
| 23     | Nf_dmd         | Demanded Fan Speed                           | Target fan speed (setpoint)             | ~1500-3000 RPM            | NO                                           |
| 24     | PCNfR_dmd      | Demanded Physical Fan Speed                  | Demanded corrected fan speed             | ~0-100%                   | NO                                           |
| 25     | W31            | HPC bleed valve position                      | Compressor bleed control                 | ~0-100%                   | MODERATE                                     |
| 26     | W32            | LPC bleed valve position                      | Compressor bleed control                 | ~0-100%                   | MODERATE                                     |

**Dataset Variants: FD001,FD002,FD003,FD004**

| Dataset | Train trajectories | Test trajectories | Conditions    | Fault Modes         |
| ------- | ------------------ | ----------------- | ------------- | ------------------- |
| FD001   | 100                | 100               | 1 (sea level) | 1 (HPC Degradation) |
| FD002   | 260                | 259               | 6             | 1                   |
| FD003   | 100                | 100               | 1             | 2 (HPC, Fan)        |
| FD004   | 248                | 249               | 6             | 2                   |

**Conceptually**
1. “Trajectories” = engines.
2. “Conditions” = different operational scenarios.
3. “Fault modes” = type of degradation that occurs.
4. **HPC** = High Pressure Compressor
5. **LPC** = Low Pressure Compressor
6. **LPT** = Low Pressure Turbine linked to HPC mechanically.

