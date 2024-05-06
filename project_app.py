import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
import joblib
import pickle  
#import plotly.figure_factory as ff
           
def main():
    
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Linear Regression','Sales Quantity Predictor'))
    
    
    #uploaded_file = st.file_uploader("Choose a file (CSV format)")
    #if uploaded_file is not None:
        #df = pd.read_csv(uploaded_file)
        #st.write('Data Preview:', df)
    def Linear_Regression():
        try:
            st.title('Linear Regression Analysis for Sales Quantity Prediction')
            st.image('cover.jpg')
            st.subheader("Select the categories below :")
            st.selectbox('Store type:', ["Grocery Store","Ecommerce","Convenience Store","Supermarket"])
    
            uploaded_file = st.file_uploader("Choose a file (CSV format)")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write('Data Preview:', df)
            df_2 = df[df['Category'] == 'Dairy']
            if st.button('Dairy Sales Quantity Analysis'):
                #filtering the data
                st.write('Dairy Data Preview:', df_2)
    
            # Allow the user to select the feature and target columns
            all_columns = df_2.columns.tolist()
            features = st.multiselect('Select feature columns:', all_columns)
            target = st.selectbox('Select target column:', all_columns)
    
            df_original = df_2.copy()
            # Keeping the Item_ID for later use
            item_ids = df_2['Item_ID'].values
    
            if st.button('Run Linear Regression'):
                # One-hot encoding
                categorical_cols = ['Item_Name', 'Category', 'Weather']
                df_2 = pd.get_dummies(df_2, columns=categorical_cols, drop_first=True)
                #st.write(features)
                # The target variable is the actual sales values
                y = df_2['Predicted_Sales'].values
                df_2.drop(columns=['Item_ID', 'Date', 'Predicted_Sales'], inplace=True)
                X = df_2.values
    
                from sklearn.preprocessing import MinMaxScaler
                # Scaling the features
                scaler_X = MinMaxScaler()
                X = scaler_X.fit_transform(X)
                # Scaling the target
                scaler_y = MinMaxScaler()
                y = scaler_y.fit_transform(y.reshape(-1, 1))
                # Train-validation split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(X_val,y_val)
                reg.score(X_val,y_val)
                y_pred = reg.predict(X_val)
    
                # Calculate evaluation metrics for Gradient Boosting Regressor
                mae_reg = mean_absolute_error(y_val, y_pred)
                mse_reg = mean_squared_error(y_val, y_pred)
                r2_reg = r2_score(y_val, y_pred)
                st.write("Linear Regression Metrics:")
                st.write(f"Mean Absolute Error (MAE): {mae_reg}")
                st.write(f"Mean Squared Error (MSE): {mse_reg}")
                st.write(f"R^2 Score: {r2_reg}")
    
                # Plotting Actual vs Predicted values
                fig, ax = plt.subplots()
                ax.scatter(y_val, y_pred)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Actual vs Predicted')
                st.pyplot(fig)
                
        except UnboundLocalError:
            pass


    def Sales_Quantity_Predictor():   
        try:
            uploaded_file = st.file_uploader("Choose a file (CSV format)")
    
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
    
                st.subheader("Select the categories below :")
                model = st.selectbox('Store type:', ["Grocery Store","Ecommerce","Convenience Store","Supermarket"])
    
                if model == "Grocery Store":
                    groc_pickle_in1 = open('./models/reg_model_grocery_store.pkl', 'rb')
                    groc_pickle_in2 = open("./models/scaler_X_grocery_store.pkl", 'rb') 
                    groc_pickle_in3 = open("./models/scaler_y_grocery_store.pkl", 'rb') 
    
                    loaded_reg = pickle.load(groc_pickle_in1)
                    loaded_scaler_X = pickle.load(groc_pickle_in2)
                    loaded_scaler_y = pickle.load(groc_pickle_in3)
    
                if model == "Ecommerce":
                    ecom_pickle_in1 = open('./models/reg_model_ecom_store.pkl', 'rb')
                    ecom_pickle_in2 = open("./models/scaler_X_ecom_store.pkl", 'rb') 
                    ecom_pickle_in3 = open("./models/scaler_y_ecom_store.pkl", 'rb')
    
                    loaded_reg = pickle.load(ecom_pickle_in1)
                    loaded_scaler_X = pickle.load(ecom_pickle_in2)
                    loaded_scaler_y = pickle.load(ecom_pickle_in3)
    
                if model == "Convenience Store":
                    con_pickle_in1 = open('./models/reg_model_convenience_store.pkl', 'rb')
                    con_pickle_in2 = open("./models/scaler_X_convenience_store.pkl", 'rb') 
                    con_pickle_in3 = open("./models/scaler_y_convenience_store.pkl", 'rb') 
    
                    loaded_reg = pickle.load(con_pickle_in1)
                    loaded_scaler_X = pickle.load(con_pickle_in2)
                    loaded_scaler_y = pickle.load(con_pickle_in3)
    
                if model == "Supermarket":
                    super_pickle_in1 = open('./models/reg_model_supermarket_store.pkl', 'rb')
                    super_pickle_in2 = open("./models/scaler_X_supermarket_store.pkl", 'rb') 
                    super_pickle_in3 = open("./models/scaler_y_supermarket_store.pkl", 'rb')
    
                    loaded_reg = pickle.load(super_pickle_in1)
                    loaded_scaler_X = pickle.load(super_pickle_in2)
                    loaded_scaler_y = pickle.load(super_pickle_in3)
    
    
                st.write('Data Preview:', df)
                df_original = df.copy()
                item_ids = df['Item_ID'].values
                #st.write(type(item_ids[0]))
                item_no = st.text_input("Item No.", value = '')
                val_2 = st.text_input("Current Availability", value ="") 
    
                item_no = int(item_no)
                val_2 = int(val_2)
    
                if item_no not in item_ids:
                    return st.write(f"Item number {item_no} does not exist.")
        #def predict_sales_for_item_reg(item_no,avail):
            # Load the saved model and scalers
            # Load the trained model 
            #pickle_in1 = open('./models/reg_model_grocery_store.pkl', 'rb')
            #pickle_in2 = open("./models/scaler_X_grocery_store.pkl", 'rb') 
            #pickle_in3 = open("./models/scaler_y_grocery_store.pkl", 'rb') 
    
    
    
            #important piece of code, verrrrrry annoying... @shreyas
            # One-hot encoding
            categorical_cols = ['Item_Name', 'Category', 'Weather']
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            #st.write(features)
            # The target variable is the actual sales values
            y = df['Predicted_Sales'].values
            df.drop(columns=['Item_ID', 'Date', 'Predicted_Sales'], inplace=True)
            X = df.values
            st.write(X.shape)
    
            from sklearn.preprocessing import MinMaxScaler
            # Scaling the features
            scaler_X = MinMaxScaler()
            X = scaler_X.fit_transform(X)
    
            item_name = df_original[df_original['Item_ID'] == item_no]['Item_Name'].iloc[0]
            item_idx = list(item_ids).index(item_no)
            latest_data = X[item_idx].reshape(1, -1)
            predicted_sales_scaled = loaded_reg.predict(latest_data)
            predicted_sales = loaded_scaler_y.inverse_transform(predicted_sales_scaled.reshape(-1, 1))[0][0]
            sales = predicted_sales-val_2
                    #return print(item_no, item_name,predicted_sales,sales)
    
                #predict_sales_for_item_reg(item_no,val_2) 
            st.subheader(f"Predicted future sales")
            col1, col2= st.columns(2)
            col1.metric(f"**Item no.{item_no}**", item_name)
            col2.metric("**Predicted Sales**", predicted_sales,sales)
            #col3.metric("**The Quantity to Increase/Decrease is**",sales,sales)

        except UnboundLocalError:
            pass

    
    if selected_box == 'Linear Regression':
        Linear_Regression()    
    if selected_box == 'Sales Quantity Predictor':
        Sales_Quantity_Predictor()

if __name__ == "__main__":
    main()
