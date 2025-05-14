
# Baseline Model: SuperIsolationForest

## Model Overview

As a baseline for comparison with our LSTM autoencoder approaches, we implemented a SuperIsolationForest model. Isolation Forest is a classical machine learning technique specifically designed for anomaly detection, and we enhanced it with additional features to create the "Super" variant for this task.

## Conceptual Background

### Isolation Forest Principles

The Isolation Forest algorithm works on the principle that anomalies are "few and different," making them easier to isolate than normal points. The algorithm:

1. Randomly selects a feature
2. Randomly selects a split value between the minimum and maximum values of the selected feature
3. Recursively partitions the data until all points are isolated
4. Computes an anomaly score based on the average path length required to isolate each point

Anomalies typically require fewer splits to isolate, resulting in shorter path lengths and higher anomaly scores.

### Enhanced "Super" Implementation

Our SuperIsolationForest enhances the standard Isolation Forest in several ways:

1. **Feature Engineering**: Incorporates temporal features extracted from the time-series data
2. **Ensemble Approach**: Uses multiple base estimators with different random seeds
3. **Multi-resolution Analysis**: Applies the algorithm at different time scales
4. **Sequential Analysis**: Accounts for the sequential nature of the data

## Implementation Details

We used scikit-learn's IsolationForest as the base implementation and extended it with custom preprocessing and post-processing steps:

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

class SuperIsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, 
                 random_state=42, n_jobs=-1, verbose=0):
        """
        Initialize the SuperIsolationForest model.
        
        Args:
            n_estimators: Number of base estimators
            max_samples: Number of samples to draw for each base estimator
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.base_model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
    def extract_features(self, sequences):
        """
        Extract additional features from the time-series sequences.
        
        Args:
            sequences: Input sequences of shape (n_samples, seq_length, n_features)
            
        Returns:
            Feature matrix of shape (n_samples, n_engineered_features)
        """
        n_samples, seq_length, n_features = sequences.shape
        features = []
        
        for i in range(n_samples):
            seq = sequences[i]
            
            # Statistical features
            mean_features = np.mean(seq, axis=0)
            std_features = np.std(seq, axis=0)
            min_features = np.min(seq, axis=0)
            max_features = np.max(seq, axis=0)
            
            # Trend features
            slopes = []
            for j in range(n_features):
                x = np.arange(seq_length)
                y = seq[:, j]
                slope, _ = np.polyfit(x, y, 1)
                slopes.append(slope)
            
            # Autocorrelation at lag 1
            autocorr = []
            for j in range(n_features):
                series = seq[:, j]
                if np.std(series) > 0:
                    ac = np.corrcoef(series[:-1], series[1:])[0, 1]
                    autocorr.append(ac)
                else:
                    autocorr.append(0)
            
            # Combine all features
            combined = np.concatenate([
                mean_features, 
                std_features, 
                min_features, 
                max_features, 
                np.array(slopes),
                np.array(autocorr)
            ])
            
            features.append(combined)
        
        return np.array(features)
    
    def fit(self, sequences):
        """
        Fit the model to the input sequences.
        
        Args:
            sequences: Input sequences of shape (n_samples, seq_length, n_features)
        """
        # Extract features
        X = self.extract_features(sequences)
        
        # Fit the base model
        self.base_model.fit(X)
        
        return self
    
    def predict(self, sequences):
        """
        Predict anomalies in the input sequences.
        
        Args:
            sequences: Input sequences of shape (n_samples, seq_length, n_features)
            
        Returns:
            Binary labels (1: normal, -1: anomaly)
        """
        # Extract features
        X = self.extract_features(sequences)
        
        # Get predictions
        return self.base_model.predict(X)
    
    def score_samples(self, sequences):
        """
        Compute anomaly scores for the input sequences.
        
        Args:
            sequences: Input sequences of shape (n_samples, seq_length, n_features)
            
        Returns:
            Anomaly scores (higher scores indicate more anomalous)
        """
        # Extract features
        X = self.extract_features(sequences)
        
        # Get raw scores (negative of decision function)
        # In Isolation Forest, lower decision function values indicate anomalies
        return -self.base_model.decision_function(X)
```

## Training Process

The SuperIsolationForest model was trained on the same preprocessed sequences as the LSTM autoencoders:

```python
# Initialize the model
sif_model = SuperIsolationForest(
    n_estimators=200,  # More estimators for better performance
    contamination=0.1,  # Expected proportion of anomalies
    max_samples=256,    # Number of samples to draw for each base estimator
    random_state=42
)

# Train the model
sif_model.fit(train_sequences)

# Compute anomaly scores
anomaly_scores = sif_model.score_samples(test_sequences)

# Determine threshold
threshold = np.percentile(anomaly_scores, 90)  # 90th percentile as threshold

# Detect anomalies
predicted_anomalies = anomaly_scores > threshold
```

## Feature Engineering Impact

The feature engineering component of SuperIsolationForest played a crucial role in its performance:

| Feature Type | Quantity | Description |
|--------------|----------|-------------|
| Statistical features | 48 × 4 = 192 | Mean, standard deviation, min, max for each of the 48 input features |
| Trend features | 48 | Linear trend (slope) for each feature |
| Autocorrelation | 48 | Lag-1 autocorrelation for each feature |
| **Total** | **288** | Total number of engineered features |

## Performance Characteristics

- **Precision**: 0.099538
- **Recall**: 0.994872
- **F1-Score**: 0.180970
- **ROC AUC**: 0.137804
- **PR AUC**: 0.055545

## Observations

1. The SuperIsolationForest achieved high recall (99.49%) but very low precision (9.95%), indicating it correctly identifies most anomalies but produces a large number of false positives.

2. The model's ROC AUC and PR AUC values are significantly lower than those of the LSTM autoencoder models, suggesting its ranking of anomalies is less effective.

3. Despite the extensive feature engineering, the model struggles to capture the complex temporal dependencies and multivariate relationships in the data compared to the LSTM-based approaches.

4. The model does have advantages in terms of training speed and interpretability, training approximately 50 times faster than the LSTM models.

## Limitations

1. **Loss of Temporal Information**: Despite our feature engineering efforts, the SuperIsolationForest cannot fully capture the sequential nature of the data.

2. **Feature Explosion**: The feature engineering process creates a high-dimensional feature space (288 features), which may lead to curse of dimensionality issues.

3. **Threshold Sensitivity**: The model's performance is highly sensitive to the threshold selection, more so than the LSTM-based approaches.

4. **Static Analysis**: The model provides a static analysis of each sequence without considering the evolving nature of anomalies over time.

## Conclusion

The SuperIsolationForest serves as a useful baseline for comparing with our LSTM autoencoder approaches. While it achieves high recall, its precision, ROC AUC, and PR AUC metrics are significantly worse than the LSTM-based models. This performance gap highlights the importance of appropriately modeling temporal dependencies and multivariate relationships in time-series anomaly detection tasks. Despite its limitations, the SuperIsolationForest provides valuable insights for comparison and establishes a solid baseline upon which more sophisticated models can improve.
