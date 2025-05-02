import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import pickle

def generate_synthetic_data(n_samples=500):
    
    subjects = ['Mathematics', 'Physics', 'Computer Science', 'Biology', 'Chemistry', 'Literature']
    topics = [
        ['Calculus', 'Algebra', 'Geometry', 'Statistics', 'Trigonometry', 'Number Theory'],
        ['Mechanics', 'Electromagnetism', 'Thermodynamics', 'Quantum Physics', 'Relativity', 'Optics'],
        ['Programming', 'Data Structures', 'Algorithms', 'Machine Learning', 'Web Development', 'Databases'],
        ['Genetics', 'Ecology', 'Cell Biology', 'Anatomy', 'Evolution', 'Microbiology'],
        ['Organic Chemistry', 'Inorganic Chemistry', 'Physical Chemistry', 'Biochemistry', 'Analytical Chemistry'],
        ['Poetry', 'Drama', 'Novel', 'Short Story', 'Literary Criticism', 'Creative Writing']
    ]
    
    data = []
    
    for _ in range(n_samples):
        subject_idx = np.random.randint(0, len(subjects))
        subject = subjects[subject_idx]
        topic = np.random.choice(topics[subject_idx])
        
        
        abstract_level = np.random.randint(1, 11)  
        prerequisites_count = np.random.randint(0, 6)  
        content_volume = np.random.randint(1, 11)  
        concept_complexity = np.random.randint(1, 11)  
        
        prereq_text = f"Requires {prerequisites_count} prerequisite topics. "
        complexity_text = f"Has conceptual complexity of {concept_complexity}/10. "
        volume_text = f"Typically takes about {content_volume} hours to cover adequately. "
        description = f"Topic: {topic} in {subject}. {prereq_text}{complexity_text}{volume_text}"
        
        if subject == 'Mathematics':
            description += np.random.choice([
                "Involves formulas and theorems. ", 
                "Requires numerical computation skills. ",
                "Contains abstract mathematical concepts. "
            ])
        elif subject == 'Physics':
            description += np.random.choice([
                "Involves laboratory experiments. ", 
                "Requires understanding of physical laws. ",
                "Applies mathematical models to physical phenomena. "
            ])
        elif subject == 'Computer Science':
            description += np.random.choice([
                "Requires coding implementation. ", 
                "Involves computational thinking. ",
                "Focuses on algorithmic efficiency. "
            ])
        
        # Calculate difficulty score with some noise
        base_difficulty = (abstract_level * 0.3 + 
                          prerequisites_count * 0.2 + 
                          content_volume * 0.15 + 
                          concept_complexity * 0.35)
        noise = np.random.normal(0, 0.5)
        difficulty = min(max(base_difficulty + noise, 1), 10)  # Keep within 1-10
        
        data.append({
            'subject': subject,
            'topic': topic,
            'description': description,
            'abstract_level': abstract_level,
            'prerequisites_count': prerequisites_count,
            'content_volume': content_volume, 
            'concept_complexity': concept_complexity,
            'difficulty': difficulty
        })
    
    return pd.DataFrame(data)

def preprocess_text(texts, tokenizer=None, max_len=100, fit=False):
    
    if fit:
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(texts)
    elif tokenizer is None:
        raise ValueError("Tokenizer must be provided if fit=False")
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, tokenizer

def preprocess_numerical(features, scaler=None, fit=False):
    
    if fit:
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
    elif scaler is None:
        raise ValueError("Scaler must be provided if fit=False")
    else:
        features_scaled = scaler.transform(features)
        
    return features_scaled, scaler

def save_preprocessors(tokenizer, scaler, tokenizer_path='tokenizer.pkl', scaler_path='scaler.pkl'):
    """Save tokenizer and scaler to disk."""
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

def load_preprocessors(tokenizer_path='tokenizer.pkl', scaler_path='scaler.pkl'):
    """Load tokenizer and scaler from disk."""
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return tokenizer, scaler

# Example of how to use these functions
if __name__ == "__main__":
    # Generate sample data
    df = generate_synthetic_data(10)
    print(df[['subject', 'topic', 'difficulty']])
    
    # Example of preprocessing
    text_data, tokenizer = preprocess_text(df['description'].values, fit=True)
    
    numerical_features = df[['abstract_level', 'prerequisites_count', 
                          'content_volume', 'concept_complexity']].values
    numerical_scaled, scaler = preprocess_numerical(numerical_features, fit=True)
    
    print(f"Text data shape: {text_data.shape}")
    print(f"Numerical features shape: {numerical_scaled.shape}")