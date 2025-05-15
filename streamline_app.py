import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Set page config
st.set_page_config(page_title="NYC Airbnb Analysis", layout="wide")

# Title
st.title("NYC Airbnb Price Analysis & Visualization")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/your-username/airbnb-price-prediction/main/AB_NYC_2019.csv"
    return pd.read_csv(url)

df = load_data()

# Data preprocessing
df['price'] = df['price'].astype(float)
df = df[['price', 'bedrooms', 'bathrooms', 'accommodates', 
         'neighbourhood_group', 'room_type']].dropna()

# Sidebar for user inputs
st.sidebar.header("User Input Features")
selected_borough = st.sidebar.selectbox(
    'Select Borough',
    df['neighbourhood_group'].unique()
)

selected_room_type = st.sidebar.selectbox(
    'Select Room Type',
    df['room_type'].unique()
)

price_range = st.sidebar.slider(
    'Price Range',
    float(df['price'].min()),
    float(df['price'].max()),
    (50.0, 300.0)
)

# Filter data based on user input
filtered_df = df[
    (df['neighbourhood_group'] == selected_borough) &
    (df['room_type'] == selected_room_type) &
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1])
]

# Display filtered data
st.subheader("Filtered Data")
st.write(f"Displaying {len(filtered_df)} listings")
st.dataframe(filtered_df.head())

# Visualization section
st.header("Data Visualizations")

# Plot 1: Price Distribution
st.subheader("Price Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df['price'], bins=50, kde=True, ax=ax1)
ax1.set_xlabel("Price ($)")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Plot 2: Price by Bedrooms
st.subheader("Price by Number of Bedrooms")
fig2, ax2 = plt.subplots()
sns.boxplot(x='bedrooms', y='price', data=filtered_df, ax=ax2)
ax2.set_xlabel("Bedrooms")
ax2.set_ylabel("Price ($)")
st.pyplot(fig2)

# Plot 3: Correlation Heatmap
st.subheader("Feature Correlation")
numerical_df = filtered_df.select_dtypes(include=['float64', 'int64'])
fig3, ax3 = plt.subplots()
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# Price Prediction Section
st.header("Price Prediction")

# User inputs for prediction
col1, col2, col3 = st.columns(3)
with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1)
with col2:
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
with col3:
    accommodates = st.number_input("Accommodates", min_value=1, max_value=20, value=2)

# Train model (simplified for demo)
@st.cache_data
def train_model(data):
    encoder = OneHotEncoder(sparse_output=False)
    categorical = data[['neighbourhood_group', 'room_type']]
    encoded = encoder.fit_transform(categorical)
    
    X = np.concatenate([
        data[['bedrooms', 'bathrooms', 'accommodates']].values,
        encoded
    ], axis=1)
    
    y = data['price'].values
    model = LinearRegression()
    model.fit(X, y)
    return model, encoder

model, encoder = train_model(df)

# Make prediction
if st.button("Predict Price"):
    input_cat = pd.DataFrame([[selected_borough, selected_room_type]], 
                           columns=['neighbourhood_group', 'room_type'])
    encoded = encoder.transform(input_cat)
    
    input_data = np.concatenate([
        [[bedrooms, bathrooms, accommodates]],
        encoded
    ], axis=1)
    
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Price: ${prediction:,.2f}")

# Data summary
st.header("Data Summary")
st.write(f"Total Listings: {len(df)}")
st.write(f"Average Price: ${df['price'].mean():,.2f}")
st.write(f"Most Expensive Borough: {df.groupby('neighbourhood_group')['price'].mean().idxmax()}")
