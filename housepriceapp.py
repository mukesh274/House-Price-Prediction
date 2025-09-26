# housepriceapp.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data():
    data = pd.read_csv("Housing.csv")

    # Convert Yes/No to 1/0
    yes_no_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in yes_no_cols:
        data[col] = data[col].map({'yes': 1, 'no': 0})

    # Convert furnishingstatus to numeric codes
    data['furnishingstatus'] = data['furnishingstatus'].astype('category').cat.codes

    return data

df = load_data()

# Split features and target
X = df.drop('price', axis=1)
y = df['price']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app layout
st.title("Housing Price Prediction App")
st.write("Enter house features below to predict the price:")

# Inputs
area = st.number_input("Area (in sq. ft.)", min_value=500, max_value=10000, step=100)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
stories = st.slider("Stories", 1, 5, 2)
mainroad = st.selectbox("Main Road Access", ['Yes', 'No'])
guestroom = st.selectbox("Guest Room", ['Yes', 'No'])
basement = st.selectbox("Basement", ['Yes', 'No'])
hotwaterheating = st.selectbox("Hot Water Heating", ['Yes', 'No'])
airconditioning = st.selectbox("Air Conditioning", ['Yes', 'No'])
parking = st.slider("Parking (Number of cars)", 0, 5, 1)
prefarea = st.selectbox("Preferred Area", ['Yes', 'No'])
furnishingstatus = st.selectbox("Furnishing Status", ['Furnished', 'Semi-furnished', 'Unfurnished'])

# Prepare input for prediction
input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [1 if mainroad == 'Yes' else 0],
    'guestroom': [1 if guestroom == 'Yes' else 0],
    'basement': [1 if basement == 'Yes' else 0],
    'hotwaterheating': [1 if hotwaterheating == 'Yes' else 0],
    'airconditioning': [1 if airconditioning == 'Yes' else 0],
    'parking': [parking],
    'prefarea': [1 if prefarea == 'Yes' else 0],
    'furnishingstatus': [0 if furnishingstatus == 'Unfurnished' else 1 if furnishingstatus == 'Semi-furnished' else 2]
})

# Predict price
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")

# Visualizations
st.subheader("Price Distribution")
fig, ax = plt.subplots()
sns.histplot(df['price'], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)
