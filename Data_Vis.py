import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="CSV Data Visualization App",
    page_icon="📊",
    layout="wide"
)

# App title
st.title("CSV Data Visualization App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Main content
if uploaded_file is not None:
    st.write("Uploaded file:")
    st.write(uploaded_file.name)

    # Read CSV file
    df = pd.read_csv(uploaded_file)

    # Show DataFrame
    st.write("Data Preview:")
    st.write(df.head())

    # Data Visualization
    st.subheader("Data Visualization")

    # Choose a visualization type
    visualization_type = st.selectbox("Select a visualization type", ["Histogram", "Pair Plot", "Correlation Heatmap"])

    if visualization_type == "Histogram":
        st.write("Columns with numerical values:")
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        selected_column = st.selectbox("Select a column for the histogram", numeric_columns)

        plt.hist(df[selected_column], bins=20, alpha=0.7)
        st.pyplot(plt)

    elif visualization_type == "Pair Plot":
        sns.pairplot(df)
        st.pyplot(plt)

    elif visualization_type == "Correlation Heatmap":
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plt)
