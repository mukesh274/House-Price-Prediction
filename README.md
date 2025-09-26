#🏡 Housing Price Prediction App

This is an interactive **Streamlit web application** that predicts housing prices based on various features such as area, number of bedrooms, bathrooms, furnishing status, and more.  
It uses a **Linear Regression model** trained on the `Housing.csv` dataset.

#🚀 Features
- 📊 Predicts house price based on user input  
- 🧮 Uses Linear Regression for price prediction  
- 📈 Interactive visualizations:
  - Price distribution histogram
  - Correlation heatmap of housing features  
- 🎛️ User-friendly interface with sliders, dropdowns, and number inputs  

---
# Install Dependencies
pip install -r requirements.txt

# If you don't have requirement.txt then install manually
pip install streamlit pandas scikit-learn matplotlib seaborn

# Run the app
streamlit run housepriceapp.py


