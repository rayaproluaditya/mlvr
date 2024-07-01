import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title of the app
st.title('Multiple Linear Regression App')

# Load data
st.sidebar.subheader('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Select the target variable
    target = st.sidebar.selectbox('Select the target variable', data.columns)

    # Select feature variables
    features = st.sidebar.multiselect('Select the feature variables', data.columns)

    if target and features:
        X = data[features]
        y = data[target]

        # Split the data
        test_size = st.sidebar.slider('Test size (percentage)', 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'Mean Squared Error: {mse}')

        # Visualize
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        st.pyplot(fig)

else:
    st.write('Please upload a CSV file to get started.')