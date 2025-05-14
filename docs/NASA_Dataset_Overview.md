# NASA SMAP & MSL Dataset Overview

## Dataset Introduction

The NASA Soil Moisture Active Passive (SMAP) and Mars Science Laboratory (MSL) datasets are specialized collections of telemetry data used for benchmarking anomaly detection in space systems. These datasets were originally published by NASA and have become standard benchmarks in time-series anomaly detection research.

## Dataset Characteristics

### SMAP Dataset
- **Source**: NASA Soil Moisture Active Passive satellite
- **Type**: Multivariate time-series data
- **Time Period**: 2015-2017
- **Variables**: Temperature, power, radiation, and various command and equipment statuses
- **Labeled Anomalies**: Contains expert-labeled anomalies representing real spacecraft issues

### MSL Dataset
- **Source**: Mars Science Laboratory (Curiosity rover)
- **Type**: Multivariate time-series data
- **Time Period**: 2012-2015
- **Variables**: Various rover subsystem telemetry readings
- **Labeled Anomalies**: Contains labeled anomalies representing actual anomalous states

## Channel Selection

For this study, we focused on the **M-6 channel** which demonstrated clearer anomaly patterns compared to other channels such as P-1. The M-6 channel represents telemetry data from critical rover systems with the following characteristics:

- **Number of Features**: 60
- **Sequence Length Used**: 100
- **Sampling Rate**: 1 reading per minute
- **Number of Training Samples**: 1,172
- **Number of Test Samples**: 1,950
- **Anomaly Percentage in Test Data**: 10.00%

## Dataset Challenges

The NASA SMAP & MSL dataset presents several unique challenges:

1. **Class Imbalance**: Anomalies represent only ~10% of the test data
2. **Complex Temporal Dependencies**: Anomalies may develop over extended periods
3. **Multivariate Relationships**: Anomalies often manifest across multiple sensor readings
4. **Seasonal and Cyclic Patterns**: Normal operation includes various repeating patterns
5. **Noise and Interference**: Space environment introduces various forms of signal noise


## Reference

B. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding," in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, pp. 387–395.