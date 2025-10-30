import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Google Data Analytics Capstone", layout="wide")

st.title("📊 Google Advanced Data Analytics Capstone Project")
st.markdown("""
### 🧠 End-to-End Machine Learning Project  
This Streamlit dashboard showcases the final capstone project from the **Google Advanced Data Analytics Professional Certificate**.  
It demonstrates **data cleaning, EDA, model training, and insights generation**.
""")

# --- Dataset Upload or Load ---
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset successfully uploaded!")
else:
    st.info("Upload your dataset to explore more.")
    data = pd.DataFrame()

# --- Display dataset ---
if not data.empty:
    st.subheader("📋 Data Preview")
    st.dataframe(data.head())

    # --- EDA Section ---
    st.subheader("📈 Exploratory Data Analysis")
    st.write("Basic insights from the dataset:")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape of data:", data.shape)
        st.write("Missing values:", data.isnull().sum().sum())

    with col2:
        st.write("Data types:")
        st.write(data.dtypes)

    # --- Visualization Section ---
    st.subheader("📊 Visualizations")
    numeric_cols = data.select_dtypes(include='number').columns.tolist()

    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Select X-axis", numeric_cols)
        y_axis = st.selectbox("Select Y-axis", numeric_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for scatter plot.")

# --- Footer ---
st.markdown("---")
st.markdown("""
✅ Created by **Vana Karthik**  
🎓 [Verified Google Capstone Certificate](https://coursera.org/verify/Z9F0WG7HRN9W)  
💻 Repository: [GitHub](https://github.com/<your-username>/Google-Advanced-Data-Analytics-Capstone)
""")
