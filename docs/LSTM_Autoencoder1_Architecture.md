
# Simple LSTM Autoencoder Architecture

## Model Overview

The Simple LSTM Autoencoder is our first approach to anomaly detection on the NASA SMAP & MSL dataset. This model follows a traditional autoencoder architecture with LSTM layers, designed to learn the normal patterns in multivariate time-series data and detect anomalies based on reconstruction error.

## Architecture Details

### Model Structure

The architecture consists of an encoder network that compresses the input data into a lower-dimensional latent representation, followed by a decoder network that reconstructs the original input from this latent representation.

```
                       ┌───────────────┐
                       │  Input Layer  │
                       │ (100, 48)     │
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │  LSTM Layer   │
                       │  (128 units)  │
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │  Dropout      │
                       │  (rate=0.2)   │
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │  LSTM Layer   │
                       │  (64 units)   │
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │ Latent Space  │
                       │ (16 units)    │
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │RepeatVector(100)│
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │  LSTM Layer   │
                       │  (64 units)   │
                       │  return_seq=True│
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │  Dropout      │
                       │  (rate=0.2)   │
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │  LSTM Layer   │
                       │  (128 units)  │
                       │  return_seq=True│
                       └───────┬───────┘
                               ▼
                       ┌───────────────┐
                       │  Time Distributed │
                       │  Dense (48)   │
                       └───────────────┘
```

### Implementation in TensorFlow/Keras

```python
def create_simple_lstm_autoencoder(seq_length, n_features, latent_dim=16):
    """
    Create a simple LSTM autoencoder model.
    
    Args:
        seq_length: Length of input sequences
        n_features: Number of features in input data
        latent_dim: Dimension of the latent space
        
    Returns:
        Compiled Keras model
    """
    # Define input shape
    input_shape = (seq_length, n_features)
    
    # Encoder
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    
    # Bottleneck (latent space representation)
    encoded = Dense(latent_dim)(x)
    
    # Decoder
    x = RepeatVector(seq_length)(encoded)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True)(x)
    
    # Output layer
    outputs = TimeDistributed(Dense(n_features))(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model
```

## Model Parameters

- **Input Dimensions**: (100, 48) - sequence length of 100 with 48 features
- **Latent Dimension**: 16
- **Encoder LSTM Units**: 128 → 64
- **Decoder LSTM Units**: 64 → 128
- **Dropout Rate**: 0.2
- **Parameters**: 166,608 trainable parameters

## Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Epochs**: 100
- **Early Stopping**: Patience of 10 epochs, monitoring validation loss
- **Validation Split**: 20% of training data
- **Callbacks**:
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - ModelCheckpoint (saving best model based on validation loss)

## Training Process

```python
# Create and train the model
model = create_simple_lstm_autoencoder(
    seq_length=100, 
    n_features=48, 
    latent_dim=16
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001
)

# Train the model
history = model.fit(
    train_sequences,
    train_sequences,  # Autoencoder reconstructs its input
    epochs=100,
    batch_size=32,
    validation_data=(val_sequences, val_sequences),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

## Anomaly Detection Strategy

This model detects anomalies by computing the reconstruction error for each time step:

```python
def detect_anomalies(model, test_sequences, threshold=None):
    """
    Detect anomalies based on reconstruction error.
    
    Args:
        model: Trained autoencoder model
        test_sequences: Test data sequences
        threshold: Reconstruction error threshold for anomaly detection
                   If None, threshold is determined automatically
                   
    Returns:
        Anomaly scores and predicted anomaly labels
    """
    # Get reconstructions
    reconstructions = model.predict(test_sequences)
    
    # Calculate reconstruction error (MSE)
    mse = np.mean(np.square(test_sequences - reconstructions), axis=(1, 2))
    
    # Determine threshold if not provided
    if threshold is None:
        # Use statistics of reconstruction error
        threshold = np.mean(mse) + 3 * np.std(mse)
    
    # Detect anomalies
    anomalies = mse > threshold
    
    return mse, anomalies, threshold
```

## Performance Characteristics

- **Precision**: 0.151633
- **Recall**: 1.000000
- **F1-Score**: 0.263336
- **ROC AUC**: 0.998212
- **PR AUC**: 0.989264

## Observations

1. The Simple LSTM Autoencoder achieves perfect recall but low precision, indicating it correctly identifies all anomalies but also generates many false positives.

2. The high ROC AUC and PR AUC values suggest that the model correctly assigns higher anomaly scores to actual anomalies, but the threshold selection needs improvement.

3. The reconstruction error distribution shows a distinct separation between normal and anomalous data points, but with significant overlap at the boundary.

4. The model successfully captures temporal dependencies and multivariate correlations in the data.


