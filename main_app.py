import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
import os
import streamlit as st
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="CSV Data Visualization App",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("App Description")
st.sidebar.write(
    "This app allows you to upload a CSV file for analysis using PandasAI and OpenAI's API. "
    "You can visualize data, get statistics, and generate insights by providing prompts."
    "by Piyush Gone"
)

load_dotenv() # To load the env file

API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm)

st.title("""
Prompt Driven Data Analysis with OpenAI
""")

uploaded_file = st.file_uploader("Upload your CSV file for analysis",type=['csv'])

if uploaded_file is not None:

    df = pd.read_csv (uploaded_file, encoding='latin1') #, errors='ignore')
    st.subheader("Data Preview:")
    st.write(df.head())

    # Show entire DataFrame
    if st.button("View Entire DataFrame"):
        st.write("Entire DataFrame:")
        st.write(df)

    # Statistics Button
    if st.button("Show Statistics"):
        st.subheader("Statistics")
        st.write(df.describe())

    # Data Visualization
    st.subheader("Data Visualization")

    # Choose a visualization type
    visualization_type = st.selectbox("Select a visualization type", ["Histogram", "Pair Plot","Pair Plot with Linear Regression", "Correlation Heatmap"])

    if visualization_type == "Histogram":
        st.write("Columns with numerical values:")
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        selected_column = st.selectbox("Select a column for the histogram", numeric_columns)

        plt.hist(df[selected_column], bins=20, alpha=0.7)
        st.pyplot(plt)

    elif visualization_type == "Pair Plot":
        sns.pairplot(df)
        st.pyplot(plt)

    elif visualization_type == "Pair Plot with Linear Regression":
        sns.pairplot(df, kind="reg")
        st.pyplot(plt)

    elif visualization_type == "Correlation Heatmap":
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)

    # Prompt to AI
    st.subheader("PandasAI")
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response ..."):
                st.write("PandasAI is generating an answer, please wait ...")
                st.write(pandas_ai.run(df,prompt=prompt))
        else:
            st.warning("Please enter a prompt.")