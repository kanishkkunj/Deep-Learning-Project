import streamlit as st
import os
os.environ['MPLBACKEND'] = 'Agg'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pickle


from data_utils import (
    generate_synthetic_data, preprocess_text, preprocess_numerical,
    save_preprocessors, load_preprocessors
)
from model import (
    build_model, train_model, predict_difficulty,
    plot_training_history, plot_factor_contribution
)


st.set_page_config(
    page_title="Topic Difficulty Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Topic Difficulty Recommender for Teachers")
st.markdown("""
    This application helps teachers estimate the difficulty level of academic topics
    based on various factors like subject area, prerequisites, complexity, and more.
    The system uses deep learning to predict difficulty scores on a scale of 1-10.
""")

# Function to train or load model
@st.cache_resource
def get_model_and_preprocessors(retrain=False):
    """
    Train or load the model and preprocessors.
    
    Args:
        retrain: Whether to retrain the model even if it exists
        
    Returns:
        model, tokenizer, scaler
    """
    model_path = 'difficulty_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    scaler_path = 'scaler.pkl'
    
    if not os.path.exists(model_path) or retrain:
        # Generate and prepare training data
        df = generate_synthetic_data(n_samples=500)
        
        # Preprocess text
        text_data, tokenizer = preprocess_text(df['description'].values, fit=True)
        
        # Preprocess numerical features
        numerical_features = df[['abstract_level', 'prerequisites_count', 
                              'content_volume', 'concept_complexity']].values
        numerical_scaled, scaler = preprocess_numerical(numerical_features, fit=True)
        
        # Train model
        model, history = train_model(
            text_data, numerical_scaled, df['difficulty'].values,
            epochs=30, batch_size=32, verbose=1
        )
        
        # Save model and preprocessors
        model.save(model_path)
        save_preprocessors(tokenizer, scaler)
        
        return model, tokenizer, scaler, history, df
    else:
        # Load existing model and preprocessors
        model = load_model(model_path)
        tokenizer, scaler = load_preprocessors()
        return model, tokenizer, scaler, None, None

# Sidebar for options
st.sidebar.title("Options")
retrain = st.sidebar.button("Retrain Model")

# Get or train model
with st.spinner("Loading/training model..."):
    if retrain:
        model, tokenizer, scaler, history, df = get_model_and_preprocessors(retrain=True)
        
        # Display training history if model was retrained
        if history:
            st.sidebar.subheader("Training Results")
            fig = plot_training_history(history)
            st.sidebar.pyplot(fig)
    else:
        model, tokenizer, scaler, _, _ = get_model_and_preprocessors()

# View sample data option
st.sidebar.subheader("Training Data")
if st.sidebar.button("View Sample Training Data"):
    sample_df = generate_synthetic_data(10)
    st.sidebar.dataframe(sample_df[['subject', 'topic', 'difficulty']].sort_values('difficulty', ascending=False))

# Main input form
st.subheader("Input Topic Details")

col1, col2 = st.columns(2)

with col1:
    subject = st.selectbox(
        "Subject Area",
        ['Mathematics', 'Physics', 'Computer Science', 'Biology', 'Chemistry', 'Literature', 'Other']
    )
    
    topic = st.text_input("Topic Name", placeholder="e.g., Neural Networks")
    
    abstract_level = st.slider(
        "Abstraction Level (1-10)",
        min_value=1,
        max_value=10,
        value=5,
        help="How abstract are the concepts? 1=Very concrete, 10=Highly abstract"
    )
    
    prereq_count = st.slider(
        "Number of Prerequisites",
        min_value=0,
        max_value=10,
        value=2,
        help="How many prerequisite topics should students know?"
    )

with col2:
    content_volume = st.slider(
        "Content Volume (hours to teach)",
        min_value=1,
        max_value=15,
        value=3,
        help="How many teaching hours would this topic require?"
    )
    
    complexity = st.slider(
        "Conceptual Complexity (1-10)",
        min_value=1,
        max_value=10,
        value=5,
        help="How complex are the ideas? 1=Simple, 10=Very complex"
    )
    
    description = st.text_area(
        "Topic Description",
        placeholder="Provide a brief description of the topic...",
        height=100
    )

# Predict button
if st.button("Predict Difficulty"):
    if not topic or not description:
        st.warning("Please enter both a topic name and description.")
    else:
        # Enhance description with form inputs for better prediction
        enhanced_description = f"Topic: {topic} in {subject}. Requires {prereq_count} prerequisite topics. "
        enhanced_description += f"Has conceptual complexity of {complexity}/10. "
        enhanced_description += f"Typically takes about {content_volume} hours to cover adequately. "
        enhanced_description += description
        
        # Preprocess inputs
        text_input, _ = preprocess_text([enhanced_description], tokenizer, max_len=100)
        numerical_input = np.array([[abstract_level, prereq_count, content_volume, complexity]])
        numerical_input_scaled, _ = preprocess_numerical(numerical_input, scaler)
        
        # Make prediction
        with st.spinner("Analyzing topic difficulty..."):
            difficulty_prediction = predict_difficulty(model, text_input, numerical_input_scaled)
            difficulty_score = float(difficulty_prediction[0][0])
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Display gauge chart for difficulty
            fig, ax = plt.subplots(figsize=(4, 0.7))
            ax.barh([0], [10], color='lightgray')
            ax.barh([0], [difficulty_score], color=plt.cm.RdYlGn_r(difficulty_score/10))
            ax.set_xlim(0, 10)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(f"### Difficulty Score: {difficulty_score:.1f}/10")
            
            # Classification based on score
            if difficulty_score < 3:
                difficulty_level = "Easy"
                color = "green"
            elif difficulty_score < 6:
                difficulty_level = "Moderate"
                color = "orange"
            else:
                difficulty_level = "Challenging"
                color = "red"
                
            st.markdown(f"<h4 style='color:{color}'>{difficulty_level}</h4>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("Recommendations")
            
            if difficulty_score < 3:
                st.markdown("""
                    - **Teaching Approach**: Direct instruction with plenty of examples
                    - **Time Allocation**: Standard time allocation should be sufficient
                    - **Practice**: Basic exercises to reinforce understanding
                    - **Assessment**: Straightforward assessment methodology
                """)
            elif difficulty_score < 6:
                st.markdown("""
                    - **Teaching Approach**: Guided discovery with scaffolded examples
                    - **Time Allocation**: Consider adding 20% more time than usual
                    - **Practice**: Mix of basic and some challenging problems
                    - **Assessment**: Include both direct and application questions
                """)
            else:
                st.markdown("""
                    - **Teaching Approach**: Project-based or flipped classroom might be effective
                    - **Time Allocation**: Allow for at least 40% more time than usual
                    - **Practice**: Extensive practice with varying difficulty levels
                    - **Assessment**: Consider multiple assessment methods
                    - **Support**: Prepare additional support materials or tutorials
                """)
            
            # Factor contribution
            st.markdown("#### Factor Contribution to Difficulty")
            fig = plot_factor_contribution(abstract_level, prereq_count, content_volume, complexity)
            st.pyplot(fig)





if __name__ == "__main__":
    # This will run when the script is executed directly
    print("Streamlit app is running!")
