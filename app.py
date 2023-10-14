import streamlit as st 
import pandas as pd
import pickle
import numpy as np
import warnings
import os 

warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")
# Initialize an empty DataFrame to store input data and predictions
# Create a widget in the sidebar to display the predicted sales

# setup
#setup
#variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "model", "ml.pkl")

#useful functions
st.cache_resource()
def  load_ml_components(fp):
    "load the ml components to re-use in app"
    with open(fp, 'rb') as file:
        obj = pickle.load(file)
        return obj
    
#Execution
ml_components_dict = load_ml_components(fp = ml_core_fp)
encoder= ml_components_dict["encoder"]

# Specify the path to your saved model
dt_model_filename = 'sales_dt_model.pkl'
sales_dt_path = os.path.join(DIRPATH, dt_model_filename)

# Load the model
model = ml_components_dict["models"]

# Set the page title and add some style
#@st.cache_resource()
st.set_page_config(
    page_title="Retail Store Sales ",
    page_icon="💰",
    layout="wide",  # Adjust layout to wide
)
# Title and description
st.write("# Retail Store Sales ")
st.write("Impact sales of a retail store.")
prediction = st.container()

# set up the sidebar
st.sidebar.header("Data Dictionary")
st.sidebar.header("Dataset")
st.sidebar.markdown("Preview the dataset used for training.")

    # Define the path to your dataset CSV file
dataset_path = 'C:/Users/manca/OneDrive/Desktop/Stream/Streamlit/dataset/sampled_train.csv'  # Replace with the actual path

    # Load the dataset into a DataFrame
dataset = pd.read_csv(dataset_path)
#Dropped columns not used in training
# Define the columns to drop
columns_to_drop = ["sales", "diff"]

# Drop the specified columns
dataset = dataset.drop(columns=columns_to_drop)

if st.sidebar.checkbox("Preview the dataset"):
        # Display the first 10 rows of the dataset
        st.sidebar.dataframe(dataset.head(10))


# Define the form 
form = st.form(key = "Information", clear_on_submit=True)

# create a key list 
expected_input= ['store_nbr', 'product', 'onpromotion', 'oil_prices', 'city','state', 'stores_type', 'cluster', 'Year', 'Month', 'Day']
numerics= ["sales","oil_prices","onpromotion","Year","Month","Day","store_nbr","cluster"]
categories= ["product","city","state","stores_type",]

# set up the prediction section 
with prediction:
    prediction.subheader("Inputs")
    prediction.write("This section will recieve inputs")
    left_col,right_col = prediction.columns(2)
#Define product list 
product_list= ['EGGS', 'HOME CARE', 'BEVERAGES', 'LIQUOR,WINE,BEER', 'DELI', 'POULTRY',
    'SCHOOL AND OFFICE SUPPLIES', 'HARDWARE', 'PRODUCE', 'BREAD/BAKERY', 'DAIRY',
    'CELEBRATION' ,'MAGAZINES' ,'PET SUPPLIES', 'PERSONAL CARE' ,'PREPARED, FOODS',
    'LADIESWEAR' ,'LAWN AND GARDEN' ,'BABY CARE' ,'GROCERY I', 'CLEANING',
    'SEAFOOD', 'MEATS', 'GROCERY II' ,'BEAUTY', 'AUTOMOTIVE', 'LINGERIE',
    'HOME APPLIANCES', 'PLAYERS AND ELECTRONICS' ,'FROZEN FOODS',
    'HOME AND KITCHEN II' ,'HOME AND KITCHEN I', 'BOOKS']
#Define city list 
city_list= ['Santo Domingo', 'Cuenca' ,'Guayaquil', 'Quito', 'Manta' ,'Ambato' ,'Quevedo',
    'Ibarra' ,'Machala', 'Riobamba' ,'Cayambe', 'Loja' ,'El Carmen', 'Daule'
    'Libertad', 'Latacunga', 'Babahoyo' ,'Salinas' ,'Esmeraldas' ,'Puyo' ,'Playas',
    'Guaranda'] 
# Define state list
state_list= ['Santo Domingo de los Tsachilas', 'Azuay' ,'Guayas' ,'Pichincha' ,'Manabi',
    'Tungurahua', 'Los Rios' ,'Imbabura' ,'El Oro' ,'Chimborazo' ,'Loja',
    'Cotopaxi', 'Santa Elena' ,'Esmeraldas' ,'Pastaza' ,'Bolivar']
# Initialize an empty DataFrame to store input data and predictions

    #set up the form 
with form:
        date = left_col.date_input("Select a Date")
          # Extract day, month, and year from the selected date
        selected_date = pd.to_datetime(date)
        Day = selected_date.day
        Month = selected_date.month
        Year = selected_date.year

       
        store_nbr = left_col.number_input("Enter store number")
        product = left_col.selectbox("Enter your product name:",options=product_list )
        onpromotion = left_col.number_input("Enter the onpromotion price")
        
        
        # set the right column 
      
        
        oil_prices = right_col.number_input("Enter the current oil prices")
        
        city = right_col.selectbox("Enter your city:",options = city_list)
        user_encoded = pd.get_dummies(pd.DataFrame({'city': [city]}), columns=['city'])
        user_encoded = user_encoded.reindex(columns=city_list, fill_value=0)
        state = right_col.selectbox("Enter your state:",options= state_list)
        stores_type = right_col.selectbox("Enter the store type:",['C', 'B' ,'D' ,'A' ,'E'])
        cluster = right_col.slider("Enter the cluster", min_value=1, value= 1)
        
        # create a submitted button
        submitted = form.form_submit_button("Submit")


# Dataframe creation
if submitted:
    with prediction:
        #formate input 
        input_dict  ={
        "store_nbr": [store_nbr],
        "product": [product],
        "onpromotion": [onpromotion],
        "oil_prices": [oil_prices],
        "city": [city],
        "state": [state],
        "stores_type": [stores_type],
        "cluster": [cluster],
        "Year": [Year],  # Set default year value
        "Month": [Month],  # Use the extracted month
        "Day": [Day], # Use the extracted day
        
        }
        
       

    # Clear the form for new input
        form.empty()
    

        # Display input data as a dataframe
        input_data = pd.DataFrame.from_dict(input_dict)
        
       # Specify the categorical columns to one-hot encode
        cat_columns = ['product', 'city', 'state', 'stores_type']
        
        # Transform test data to one-hot encoded format for the specified columns
        encoded_sdata = encoder.transform(input_data[cat_columns])

        # Get the feature names from the encoder
        feature_names = encoder.get_feature_names_out(input_features=cat_columns)
        
        # Create a DataFrame using the encoded data and feature names
        encoded_tcat = pd.DataFrame(encoded_sdata.toarray(), columns=feature_names)

        # Concatenate the one-hot encoded DataFrame and the remaining columns
        # Get a list of columns that were not one-hot encoded
        remaining_columns = [col for col in input_data.columns if col not in cat_columns]
        
        # Concatenate the one-hot encoded DataFrame and the remaining columns using the 'concat' method
        encoded_test = pd.concat([encoded_tcat, input_data[remaining_columns]], axis=1)

       
    # Call the loaded model to make predictions
        model_output = model.predict(encoded_test) # Use the loaded model for prediction
        
   # Use the same encoded_input_data as in your code
        input_data["Prediction"] = model_output
        predicated_sales = model_output
        #
# Display the prediction with a success message
        st.success(f"Prediction of Sales: {predicated_sales}")
    #st.write("Predicted Sales:", model_output)
            #Make the prediction
      # Assuming model_output is a dictionary

    
        