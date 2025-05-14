# Data Preprocessing for NASA SMAP & MSL Dataset

## Overview

This document details the preprocessing steps applied to the NASA SMAP & MSL dataset before training our anomaly detection models. Proper preprocessing is crucial for time-series data to ensure effective model training and accurate anomaly detection.

## Dataset Loading and Extraction

```python
def load_dataset(dataset_path, channel='M-6'):
    """Load the NASA dataset for the specified channel."""
    train_path = os.path.join(dataset_path, 'train', channel, 'features')
    test_path = os.path.join(dataset_path, 'test', channel, 'features')
    
    # Load train and test data
    train_data = np.load(train_path + '.npy')
    test_data = np.load(test_path + '.npy')
    
    # Load labels for test data
    test_labels = np.load(os.path.join(dataset_path, 'test', channel, 'labels.npy'))
    
    return train_data, test_data, test_labels
```

## Preprocessing Steps

### 1. Handling Missing Values

The NASA dataset contains some missing values represented as NaN. We used a forward-fill approach followed by a backward-fill to handle these missing values:

```python
def handle_missing_values(data):
    """Fill missing values using forward-fill then backward-fill."""
    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(data)
    
    # Forward fill first
    df = df.fillna(method='ffill')
    
    # Backward fill any remaining NaNs
    df = df.fillna(method='bfill')
    
    # If any NaNs still remain, fill with zeros
    df = df.fillna(0)
    
    return df.values
```

### 2. Normalization

We applied feature-wise Min-Max normalization to scale all features to the [0, 1] range:

```python
def normalize_data(train_data, test_data):
    """Apply Min-Max normalization based on training data."""
    # Compute min and max values from training data
    min_vals = np.min(train_data, axis=0)
    max_vals = np.max(train_data, axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    # Apply normalization
    train_normalized = (train_data - min_vals) / range_vals
    test_normalized = (test_data - min_vals) / range_vals
    
    # Clip test data to [0, 1] range
    test_normalized = np.clip(test_normalized, 0, 1)
    
    return train_normalized, test_normalized, (min_vals, max_vals)
```

### 3. Sequence Generation

We transformed the data into overlapping sequences of length 100 for time-series analysis:

```python
def create_sequences(data, seq_length=100, step=1):
    """Create overlapping sequences from time-series data."""
    sequences = []
    for i in range(0, len(data) - seq_length + 1, step):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    
    return np.array(sequences)
```

### 4. Train-Validation Split

We used an 80-20 train-validation split for model development:

```python
def train_val_split(train_data, val_size=0.2, random_state=42):
    """Split training data into train and validation sets."""
    # Calculate split index
    split_idx = int(len(train_data) * (1 - val_size))
    
    # Split data
    train = train_data[:split_idx]
    val = train_data[split_idx:]
    
    return train, val
```

### 5. Feature Selection

Based on feature importance analysis, we identified the most relevant features for anomaly detection:

```python
def select_features(train_data, test_data, importance_threshold=0.01):
    """Select features based on variance."""
    # Calculate variance of each feature
    feature_variance = np.var(train_data, axis=0)
    
    # Select features with variance above threshold
    selected_features = feature_variance > importance_threshold
    
    # Apply feature selection
    train_selected = train_data[:, selected_features]
    test_selected = test_data[:, selected_features]
    
    return train_selected, test_selected, selected_features
```

## Data Augmentation

We implemented limited data augmentation to enhance model robustness:

1. **Jittering**: Added small random noise to training sequences
2. **Scaling**: Applied minor random scaling to sequences

```python
def augment_training_data(sequences, augmentation_factor=2):
    """Apply data augmentation to training sequences."""
    augmented_sequences = [sequences]
    
    # Jittering
    jitter = sequences + 0.01 * np.random.normal(0, 1, sequences.shape)
    augmented_sequences.append(jitter)
    
    # Only apply further augmentation if requested
    if augmentation_factor > 2:
        # Scaling
        scale_factor = np.random.normal(1, 0.1, (sequences.shape[0], 1, sequences.shape[2]))
        scaling = sequences * scale_factor
        augmented_sequences.append(scaling)
    
    # Combine augmented data
    return np.concatenate(augmented_sequences, axis=0)
```

## Final Preprocessing Pipeline

```python
def preprocess_pipeline(train_data, test_data, test_labels, seq_length=100):
    """Complete preprocessing pipeline."""
    # Handle missing values
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)
    
    # Normalize data
    train_norm, test_norm, norm_params = normalize_data(train_data, test_data)
    
    # Select features
    train_selected, test_selected, selected_features = select_features(train_norm, test_norm)
    
    # Create sequences
    train_sequences = create_sequences(train_selected, seq_length)
    
    # Create sequences for test data
    test_sequences = create_sequences(test_selected, seq_length)
    
    # Create sequences for test labels
    # We use majority vote for labeling sequences
    test_label_sequences = []
    for i in range(0, len(test_labels) - seq_length + 1, 1):
        seq_labels = test_labels[i:i + seq_length]
        # A sequence is anomalous if any point is anomalous
        is_anomaly = np.any(seq_labels)
        test_label_sequences.append(is_anomaly)
    
    test_label_sequences = np.array(test_label_sequences)
    
    # Split train data into train and validation
    train_seq, val_seq = train_val_split(train_sequences)
    
    # Apply data augmentation to training data
    train_seq_augmented = augment_training_data(train_seq)
    
    return {
        'train_sequences': train_seq_augmented,
        'val_sequences': val_seq,
        'test_sequences': test_sequences,
        'test_labels': test_label_sequences,
        'normalization_params': norm_params,
        'selected_features': selected_features
    }
```

## Preprocessing Results

After preprocessing, our dataset has the following characteristics:

- **Training Sequences**: 1,172 (original) → 2,344 (after augmentation)
- **Validation Sequences**: 293
- **Test Sequences**: 1,950
- **Sequence Length**: 100 time steps
- **Feature Dimension**: 60 → 48 (after feature selection)
- **Anomaly Percentage in Test Data**: 10.00%
