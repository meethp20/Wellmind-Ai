import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Mental Health ML Predictor",
    page_icon="🧠",
    layout="wide"
)

# Data preprocessing function
@st.cache_data
def preprocess_data():
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    
    # Remove the patient number column
    df = df.drop('Patient Number', axis=1)
    
    # Separate features and target
    X = df.drop('Expert Diagnose', axis=1)
    y = df['Expert Diagnose']
    
    # Initialize label encoders
    label_encoders = {}
    
    # Encode categorical features
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoders, target_encoder, df

# Train all ML models
@st.cache_data
def train_models(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'actual': y_test
        }
    
    return trained_models, results, X_test, y_test

# Main app
def main():
    st.title("🧠 Mental Health Disorder Prediction System")
    st.markdown("### AI-Powered Mental Health Diagnosis Predictor")
    st.markdown("---")
    
    # Load and preprocess data
    X, y, label_encoders, target_encoder, original_df = preprocess_data()
    
    # Train models
    trained_models, results, X_test, y_test = train_models(X, y)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model Performance", "Make Prediction", "Dataset Info"])
    
    if page == "Model Performance":
        st.header("📊 Model Performance Comparison")
        
        # Create performance dataframe
        performance_data = []
        for name, result in results.items():
            performance_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Percentage': f"{result['accuracy']*100:.2f}%"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        st.success(f"🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Detailed results for best model
        st.header(f"📈 Detailed Results - {best_model_name}")
        
        best_predictions = results[best_model_name]['predictions']
        actual_labels = target_encoder.inverse_transform(y_test)
        predicted_labels = target_encoder.inverse_transform(best_predictions)
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(actual_labels, predicted_labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(actual_labels, predicted_labels)
        cm_df = pd.DataFrame(cm, 
                            index=target_encoder.classes_, 
                            columns=target_encoder.classes_)
        st.dataframe(cm_df, use_container_width=True)
    
    elif page == "Make Prediction":
        st.header("🔮 Make a Prediction")
        st.markdown("Enter patient symptoms to get a mental health diagnosis prediction:")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        
        # Create input fields
        col1, col2 = st.columns(2)
        
        with col1:
            sadness = st.selectbox("Sadness", ['Seldom', 'Sometimes', 'Usually', 'Most-Often'])
            euphoric = st.selectbox("Euphoric", ['Seldom', 'Sometimes', 'Usually', 'Most-Often'])
            exhausted = st.selectbox("Exhausted", ['Seldom', 'Sometimes', 'Usually', 'Most-Often'])
            sleep_disorder = st.selectbox("Sleep Disorder", ['Seldom', 'Sometimes', 'Usually', 'Most-Often'])
            mood_swing = st.selectbox("Mood Swing", ['YES', 'NO'])
            suicidal_thoughts = st.selectbox("Suicidal Thoughts", ['YES', 'NO'])
            anorexia = st.selectbox("Anorexia", ['YES', 'NO'])
            authority_respect = st.selectbox("Authority Respect", ['YES', 'NO'])
            try_explanation = st.selectbox("Try Explanation", ['YES', 'NO'])
        
        with col2:
            aggressive_response = st.selectbox("Aggressive Response", ['YES', 'NO'])
            ignore_move_on = st.selectbox("Ignore & Move-On", ['YES', 'NO'])
            nervous_breakdown = st.selectbox("Nervous Break-down", ['YES', 'NO'])
            admit_mistakes = st.selectbox("Admit Mistakes", ['YES', 'NO'])
            overthinking = st.selectbox("Overthinking", ['YES', 'NO'])
            sexual_activity = st.selectbox("Sexual Activity", ['1 From 10', '2 From 10', '3 From 10', '4 From 10', '5 From 10', '6 From 10', '7 From 10', '8 From 10', '9 From 10'])
            concentration = st.selectbox("Concentration", ['1 From 10', '2 From 10', '3 From 10', '4 From 10', '5 From 10', '6 From 10', '7 From 10', '8 From 10'])
            optimism = st.selectbox("Optimism", ['1 From 10', '2 From 10', '3 From 10', '4 From 10', '5 From 10', '6 From 10', '7 From 10', '8 From 10', '9 From 10'])
        
        if st.button("🎯 Predict Mental Health Status", type="primary"):
            # Prepare input data
            input_data = [sadness, euphoric, exhausted, sleep_disorder, mood_swing, 
                         suicidal_thoughts, anorexia, authority_respect, try_explanation,
                         aggressive_response, ignore_move_on, nervous_breakdown, 
                         admit_mistakes, overthinking, sexual_activity, concentration, optimism]
            
            # Encode input data
            encoded_input = []
            feature_names = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder', 'Mood Swing',
                            'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation',
                            'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down',
                            'Admit Mistakes', 'Overthinking', 'Sexual Activity', 'Concentration', 'Optimisim']
            
            for i, value in enumerate(input_data):
                try:
                    encoded_value = label_encoders[feature_names[i]].transform([value])[0]
                    encoded_input.append(encoded_value)
                except:
                    encoded_input.append(0)  # Default value if encoding fails
            
            # Make prediction with best model
            prediction = trained_models[best_model_name].predict([encoded_input])[0]
            predicted_diagnosis = target_encoder.inverse_transform([prediction])[0]
            
            # Display prediction
            st.success(f"🎯 Predicted Diagnosis: **{predicted_diagnosis}**")
            
            # Get prediction probabilities
            if hasattr(trained_models[best_model_name], 'predict_proba'):
                probabilities = trained_models[best_model_name].predict_proba([encoded_input])[0]
                
                st.subheader("📊 Prediction Probabilities")
                prob_data = []
                for i, prob in enumerate(probabilities):
                    diagnosis = target_encoder.inverse_transform([i])[0]
                    prob_data.append({
                        'Diagnosis': diagnosis,
                        'Probability': f"{prob:.4f}",
                        'Percentage': f"{prob*100:.2f}%"
                    })
                
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, use_container_width=True)
    
    else:  # Dataset Info
        st.header("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(original_df))
        
        with col2:
            st.metric("Features", len(original_df.columns) - 2)  # Excluding Patient Number and Expert Diagnose
        
        with col3:
            st.metric("Classes", len(original_df['Expert Diagnose'].unique()))
        
        st.subheader("Target Distribution")
        target_counts = original_df['Expert Diagnose'].value_counts()
        st.bar_chart(target_counts)
        
        st.subheader("Sample Data")
        st.dataframe(original_df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.write("Target class distribution:")
        for diagnosis, count in target_counts.items():
            percentage = (count / len(original_df)) * 100
            st.write(f"- {diagnosis}: {count} samples ({percentage:.1f}%)")

if __name__ == "__main__":
    main()