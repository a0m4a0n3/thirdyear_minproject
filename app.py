import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Grocery Stock Level Prediction")

# Custom CSS for button styling and background color
st.markdown(
    """
    <style>
    body {
        background-color: #1c2e4f; /* Set background color */
        color: white; /* Default text color */
    }
    .stButton>button {
        background-color: #007bff; /* Default button color */
        color: white; /* Text color */
        border: none; /* Remove border */
        padding: 10px 20px; /* Button padding */
        border-radius: 5px; /* Rounded corners */
        cursor: pointer; /* Pointer cursor */
        width: 100%; /* Full width button */
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Change button color on hover */
        color: white; /* Maintain text color on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the data
data = pd.read_csv('data (1).csv')

# Initialize session state variables
if 'show_user_input' not in st.session_state:
    st.session_state.show_user_input = False
if 'analyze' not in st.session_state:
    st.session_state.analyze = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = pd.DataFrame(columns=['Quantity', 'Temperature', 'Unit Price', 
                                                       'Day of Month', 'Day of Week', 
                                                       'Hour', 'Category', 'Estimated Stock Percentage'])

# Input field names
input_names = ['quantity', 'temperature', 'unit_price', 
               'timestamp_day_of_month', 'timestamp_day_of_week', 'timestamp_hour', 'category']

for input_name in input_names:
    if input_name not in st.session_state:
        st.session_state[input_name] = ""

# Sidebar header
st.sidebar.header("Controls")

# Create two columns for buttons in the sidebar
col1, col2 = st.sidebar.columns(2)

# User Input Button in the first column
with col1:
    user_input_clicked = st.button("User Input")

# Analyze Button in the second column
with col2:
    analyze_clicked = st.button("Analysis")

# If User Input button is clicked, toggle input fields
if user_input_clicked:
    st.session_state.show_user_input = not st.session_state.show_user_input

# Show or hide the user input fields based on the toggle state
if st.session_state.show_user_input:
    st.sidebar.subheader("User Input")
    
    # User Input Fields
    st.session_state.quantity = st.sidebar.number_input("Quantity:", min_value=0, value=1, step=1)
    st.session_state.temperature = st.sidebar.number_input("Temperature:", min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
    
    # Category Selectbox
    categories = sorted(data['category'].unique())
    st.session_state.category = st.sidebar.selectbox("Category:", categories)
    
    # Filter unit prices based on the selected category
    filtered_unit_prices = sorted(data[data['category'] == st.session_state.category]['unit_price'].unique())
    
    # Unit Price Selectbox
    st.session_state.unit_price = st.sidebar.selectbox("Unit Price:", filtered_unit_prices)
    
    # Timestamp Inputs
    st.session_state.timestamp_day_of_month = st.sidebar.number_input("Day of Month:", min_value=1, max_value=31, value=1, step=1)
    st.session_state.timestamp_day_of_week = st.sidebar.selectbox("Day of Week:", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    st.session_state.timestamp_hour = st.sidebar.number_input("Hour of Day:", min_value=0, max_value=23, value=12, step=1)

    # Submit button for prediction
    if st.sidebar.button("Submit"):
        # Convert categorical day_of_week into a numeric format
        day_of_week_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

        # Preprocessing
        data['timestamp_day_of_week'] = data['timestamp_day_of_week'].map(day_of_week_mapping)

        # One-hot encode the selected category
        data = pd.get_dummies(data, columns=['category'], drop_first=True)

        # Features and target
        X = data.drop(['estimated_stock_pct'], axis=1)  # Assuming 'estimated_stock_pct' is the target
        y = data['estimated_stock_pct']

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prepare the user inputs for prediction
        user_input_data = {
            'quantity': st.session_state.quantity,
            'temperature': st.session_state.temperature,
            'unit_price': st.session_state.unit_price,
            'timestamp_day_of_month': st.session_state.timestamp_day_of_month,
            'timestamp_day_of_week': day_of_week_mapping[st.session_state.timestamp_day_of_week],
            'timestamp_hour': st.session_state.timestamp_hour
        }

        # One-hot encoding for the user-selected category
        category_columns = [f'category_{cat}' for cat in categories]
        for category in category_columns:
            user_input_data[category] = 1 if category == f'category_{st.session_state.category}' else 0

        # Convert user input to DataFrame
        user_input_df = pd.DataFrame([user_input_data])

        # Ensure the columns match the training data
        user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)

        # Predict the stock percentage for the user input
        predicted_stock_pct = model.predict(user_input_df)[0]

        # Store the input data along with the predicted stock percentage
        input_record = {
            'Quantity': st.session_state.quantity,
            'Temperature': st.session_state.temperature,
            'Unit Price': st.session_state.unit_price,
            'Day of Month': st.session_state.timestamp_day_of_month,
            'Day of Week': st.session_state.timestamp_day_of_week,
            'Hour': st.session_state.timestamp_hour,
            'Category': st.session_state.category,
            'Estimated Stock Percentage': predicted_stock_pct
        }

        # Create a DataFrame from the input record
        input_record_df = pd.DataFrame([input_record])

        # Ensure only relevant columns are included before concatenation
        input_record_df = input_record_df.loc[:, input_record_df.columns.intersection(st.session_state.input_data.columns)]

        # Append the new record to the existing DataFrame
        st.session_state.input_data = pd.concat([st.session_state.input_data, input_record_df], ignore_index=True)

        # Display the prediction in a message box for attraction purpose
        st.success(f"✨ Estimated Stock Percentage: **{predicted_stock_pct:.2f}** ✨")

        # Display the input data table
        st.subheader("Input Data Table")
        st.write(st.session_state.input_data)

# Handle the Analyze button click
if analyze_clicked:
    st.session_state.analyze = not st.session_state.analyze

# Display analysis results if the analyze button is active
if st.session_state.analyze:
    results = []
    
    if st.session_state.quantity:
        results.append(f"Total Quantity: {st.session_state.quantity}")
    if st.session_state.temperature:
        results.append(f"Temperature: {st.session_state.temperature}°C")
    if st.session_state.unit_price:
        revenue = st.session_state.quantity * st.session_state.unit_price
        results.append(f"Total Revenue: ${revenue:.2f}")
        
    st.subheader("Analysis Results:")
    for result in results:
        st.write(result)
