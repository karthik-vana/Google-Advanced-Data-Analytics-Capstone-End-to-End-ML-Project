import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Employee Attrition Prediction")
st.markdown("""
This application predicts whether an employee will leave the company based on various factors.
Upload your CSV data or use the sample data to get predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose an option:", 
                          ["Data Overview", "Data Analysis", "Model Training", "Predictions"])

# Sample data generation function (similar to the notebook)
def generate_sample_data():
    """Generate sample employee data similar to the notebook"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'satisfaction_level': np.random.uniform(0.1, 1.0, n_samples),
        'last_evaluation': np.random.uniform(0.3, 1.0, n_samples),
        'number_project': np.random.randint(2, 8, n_samples),
        'average_montly_hours': np.random.randint(120, 310, n_samples),
        'time_spend_company': np.random.randint(1, 11, n_samples),
        'Work_accident': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'promotion_last_5years': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'department': np.random.choice(['sales', 'technical', 'support', 'IT', 'product_mng', 
                                      'marketing', 'RandD', 'accounting', 'hr', 'management'], n_samples),
        'salary': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.5, 0.4, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on patterns (similar to real attrition)
    attrition_prob = (
        (1 - df['satisfaction_level']) * 0.3 +
        (df['time_spend_company'] > 5) * 0.2 +
        (df['average_montly_hours'] > 250) * 0.2 +
        (df['number_project'] > 6) * 0.15 +
        (df['last_evaluation'] < 0.5) * 0.15
    )
    
    df['left'] = np.random.binomial(1, attrition_prob)
    
    return df

# Data preprocessing function
def preprocess_data(df):
    """Preprocess the data similar to the notebook approach"""
    df_processed = df.copy()
    
    # Convert categorical variables
    if 'salary' in df_processed.columns:
        salary_map = {'low': 0, 'medium': 1, 'high': 2}
        df_processed['salary'] = df_processed['salary'].map(salary_map)
    
    if 'department' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['department'], prefix='dept')
    
    return df_processed

# Model training function
def train_model(X_train, y_train, model_type='random_forest'):
    """Train the selected model"""
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42)
    
    model.fit(X_train, y_train)
    return model

# Main application logic
if options == "Data Overview":
    st.header("📋 Data Overview")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    else:
        st.info("Using sample data. Upload a CSV file to use your own data.")
        df = generate_sample_data()
    
    # Display data
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Employees", len(df))
    
    with col2:
        if 'left' in df.columns:
            attrition_rate = df['left'].mean() * 100
            st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    
    with col3:
        st.metric("Number of Features", len(df.columns) - 1)  # excluding target
    
    st.subheader("Data Types")
    st.write(df.dtypes)

elif options == "Data Analysis":
    st.header("📈 Data Analysis")
    
    # Load data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="analysis")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = generate_sample_data()
    
    if 'left' not in df.columns:
        st.error("Target column 'left' not found in the dataset!")
        st.stop()
    
    # Visualization options
    st.subheader("Attrition Analysis")
    
    # Attrition distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    attrition_counts = df['left'].value_counts()
    ax[0].pie(attrition_counts.values, labels=['Stayed', 'Left'], autopct='%1.1f%%', startangle=90)
    ax[0].set_title('Employee Attrition Distribution')
    
    # Bar plot
    attrition_counts.plot(kind='bar', ax=ax[1], color=['skyblue', 'salmon'])
    ax[1].set_title('Employee Attrition Count')
    ax[1].set_xlabel('Left Company')
    ax[1].set_ylabel('Count')
    ax[1].set_xticklabels(['Stayed', 'Left'], rotation=0)
    
    st.pyplot(fig)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Select feature for analysis
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'left' in numeric_features:
        numeric_features.remove('left')
    
    selected_feature = st.selectbox("Select feature to analyze:", numeric_features)
    
    if selected_feature:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot
        df.boxplot(column=selected_feature, by='left', ax=ax)
        ax.set_title(f'{selected_feature} vs Attrition')
        ax.set_xlabel('Left Company')
        ax.set_ylabel(selected_feature)
        
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)

elif options == "Model Training":
    st.header("🤖 Model Training")
    
    # Load data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="training")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = generate_sample_data()
    
    if 'left' not in df.columns:
        st.error("Target column 'left' not found in the dataset!")
        st.stop()
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.drop('left', axis=1)
    y = df_processed['left']
    
    # Train-test split
    test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model selection
    model_type = st.selectbox("Select Model:", ["Random Forest", "Logistic Regression"])
    
    # Train model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if model_type == "Random Forest":
                model = train_model(X_train_scaled, y_train, 'random_forest')
            else:
                model = train_model(X_train_scaled, y_train, 'logistic_regression')
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            
            with col2:
                st.metric("Training Samples", len(X_train))
            
            with col3:
                st.metric("Test Samples", len(X_test))
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Feature importance (for Random Forest)
            if model_type == "Random Forest":
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
                ax.set_title('Top 10 Feature Importances')
                st.pyplot(fig)
            
            # Save model for predictions
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = X.columns.tolist()

elif options == "Predictions":
    st.header("🔮 Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' section!")
        st.stop()
    
    # Load necessary objects from session state
    model = st.session_state['model']
    scaler = st.session_state['scaler']
    feature_names = st.session_state['feature_names']
    
    # Prediction options
    prediction_option = st.radio("Choose prediction method:", 
                                ["Single Prediction", "Batch Prediction (CSV)"])
    
    if prediction_option == "Single Prediction":
        st.subheader("Single Employee Prediction")
        
        # Create input form based on feature names
        input_data = {}
        
        # Group features for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            for i, feature in enumerate(feature_names[:len(feature_names)//2]):
                if 'dept' in feature:
                    # Department features are binary
                    input_data[feature] = st.selectbox(feature, [0, 1])
                elif feature == 'salary':
                    input_data[feature] = st.selectbox("Salary Level", [0, 1, 2], 
                                                     format_func=lambda x: ['Low', 'Medium', 'High'][x])
                else:
                    # Numeric features
                    if 'satisfaction' in feature or 'evaluation' in feature:
                        input_data[feature] = st.slider(feature, 0.0, 1.0, 0.5, 0.1)
                    elif 'hours' in feature:
                        input_data[feature] = st.slider(feature, 100, 350, 200)
                    else:
                        input_data[feature] = st.number_input(feature, value=0)
        
        with col2:
            for i, feature in enumerate(feature_names[len(feature_names)//2:]):
                if 'dept' in feature:
                    input_data[feature] = st.selectbox(feature, [0, 1])
                elif feature == 'salary':
                    input_data[feature] = st.selectbox("Salary Level", [0, 1, 2], 
                                                     format_func=lambda x: ['Low', 'Medium', 'High'][x])
                else:
                    if 'satisfaction' in feature or 'evaluation' in feature:
                        input_data[feature] = st.slider(feature, 0.0, 1.0, 0.5, 0.1)
                    elif 'hours' in feature:
                        input_data[feature] = st.slider(feature, 100, 350, 200)
                    else:
                        input_data[feature] = st.number_input(feature, value=0)
        
        if st.button("Predict Attrition"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features are present and in correct order
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            input_df = input_df[feature_names]
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"🚨 High Risk: Employee is likely to leave (Probability: {probability[1]:.2%})")
            else:
                st.success(f"✅ Low Risk: Employee is likely to stay (Probability: {probability[0]:.2%})")
            
            # Probability breakdown
            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.bar(['Stay', 'Leave'], probability, color=['green', 'red'])
            ax.set_ylabel('Probability')
            ax.set_title('Attrition Probability')
            for bar, prob in zip(bars, probability):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{prob:.1%}', ha='center', va='bottom')
            st.pyplot(fig)
    
    else:  # Batch Prediction
        st.subheader("Batch Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="batch_pred")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            
            # Preprocess the batch data
            batch_processed = preprocess_data(batch_df)
            
            # Ensure all features are present
            for feature in feature_names:
                if feature not in batch_processed.columns:
                    batch_processed[feature] = 0
            
            # Select only the features used in training
            X_batch = batch_processed[feature_names]
            
            # Scale features
            X_batch_scaled = scaler.transform(X_batch)
            
            # Make predictions
            predictions = model.predict(X_batch_scaled)
            probabilities = model.predict_proba(X_batch_scaled)
            
            # Add predictions to dataframe
            results_df = batch_df.copy()
            results_df['Prediction'] = predictions
            results_df['Stay_Probability'] = probabilities[:, 0]
            results_df['Leave_Probability'] = probabilities[:, 1]
            results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'Stay', 1: 'Leave'})
            
            # Display results
            st.subheader("Prediction Results")
            st.dataframe(results_df)
            
            # Summary statistics
            st.subheader("Prediction Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_employees = len(results_df)
                st.metric("Total Employees", total_employees)
            
            with col2:
                predicted_leavers = results_df['Prediction'].sum()
                st.metric("Predicted to Leave", predicted_leavers)
            
            with col3:
                attrition_rate = (predicted_leavers / total_employees) * 100
                st.metric("Predicted Attrition Rate", f"{attrition_rate:.1f}%")
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="employee_attrition_predictions.csv",
                mime="text/csv"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application uses machine learning to predict employee attrition. "
    "It supports both Random Forest and Logistic Regression models."
)

# -- Footer --
st.markdown("""
--
✅ Created by **Vana Karthik**  
🎓 [Verified Google Capstone Certificate](https://coursera.org/verify/Z9F0WG7HRN9W)  
🔗 [LinkedIn](https://www.linkedin.com/in/karthik-vana/)  
📘 Repository: [GitHub](https://github.com/karthik-vana/Google-Advanced-Data-Analytics-Capstone-End-to-End-ML-Project)
--
""")
