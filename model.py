import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Flatten, Concatenate, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def build_model(vocab_size=5000, max_len=100, embedding_dim=50):
    """
    Build a dual-input neural network for topic difficulty prediction.
    
    Args:
        vocab_size: Size of vocabulary for embedding layer
        max_len: Maximum sequence length
        embedding_dim: Dimension of word embeddings
        
    Returns:
        Compiled Keras model
    """
    # Text input branch
    text_input = Input(shape=(max_len,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    
    # You can choose between these two architectures:
    
    # Option 1: Simple flatten (faster training)
    flatten = Flatten()(embedding)
    text_features = Dense(32, activation='relu')(flatten)
    
    # Option 2: LSTM for sequence understanding (better but slower)
    # lstm = LSTM(32)(embedding)
    # text_features = Dense(32, activation='relu')(lstm)
    
    # Numerical features branch
    numeric_input = Input(shape=(4,))
    numeric_features = Dense(16, activation='relu')(numeric_input)
    
    # Combine branches
    combined = Concatenate()([text_features, numeric_features])
    
    # Deeper layers for better feature extraction
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    # Output layer (linear activation for regression)
    output = Dense(1, activation='linear')(x)
    
    # Build model
    model = Model(inputs=[text_input, numeric_input], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    
    return model

def train_model(X_text, X_num, y, epochs=30, batch_size=32, validation_split=0.2,
                verbose=1, early_stopping=True):
    """
    Train the model with the given data.
    
    Args:
        X_text: Preprocessed text data
        X_num: Preprocessed numerical features
        y: Target difficulty scores
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Portion of data to use for validation
        verbose: Verbosity level
        early_stopping: Whether to use early stopping
        
    Returns:
        Trained model and training history
    """
    # Split data
    X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_text, X_num, y, test_size=validation_split, random_state=42
    )
    
    # Build model
    model = build_model(vocab_size=5000, max_len=X_text.shape[1])
    
    # Define callbacks
    callbacks = []
    if early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        callbacks.append(early_stop)
    
    # Train model
    history = model.fit(
        [X_text_train, X_num_train], y_train,
        validation_data=([X_text_val, X_num_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )
    
    return model, history

def evaluate_model(model, X_text_test, X_num_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_text_test: Test text data
        X_num_test: Test numerical features
        y_test: Test target values
        
    Returns:
        Evaluation metrics
    """
    results = model.evaluate([X_text_test, X_num_test], y_test, verbose=0)
    return {'mse': results[0], 'mae': results[1]}

def predict_difficulty(model, X_text, X_num):
    """
    Make predictions with the model.
    
    Args:
        model: Trained model
        X_text: Preprocessed text input
        X_num: Preprocessed numerical input
        
    Returns:
        Predicted difficulty scores
    """
    predictions = model.predict([X_text, X_num])
    # Clamp predictions to valid range (1-10)
    predictions = np.clip(predictions, 1, 10)
    return predictions

def plot_training_history(history):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history from model.fit()
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_factor_contribution(abstract_level, prereq_count, content_volume, complexity):
    """
    Plot contribution of different factors to difficulty prediction.
    
    Args:
        abstract_level: Abstraction level value
        prereq_count: Prerequisites count
        content_volume: Content volume value
        complexity: Conceptual complexity value
        
    Returns:
        Matplotlib figure
    """
    factors = {
        "Abstraction Level": abstract_level * 0.3,
        "Prerequisites": prereq_count * 0.2,
        "Content Volume": content_volume * 0.15,
        "Conceptual Complexity": complexity * 0.35
    }
    
    fig, ax = plt.subplots(figsize=(4, 3))
    y_pos = np.arange(len(factors))
    factor_values = list(factors.values())
    normalized_values = [v / sum(factor_values) * 10 for v in factor_values]
    
    ax.barh(y_pos, normalized_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(factors.keys()))
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Contribution to Difficulty (normalized)')
    ax.set_xlim(0, max(normalized_values) * 1.1)
    plt.tight_layout()
    
    return fig