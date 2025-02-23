import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import joblib
import warnings
import os

# Model parameters
image_size = (150, 150)  # Input size expected by the model
ckpt_path = "efficient_net_vehicle.ckpt"  # Path to the checkpoint

def build_model():
    """Builds the model architecture with EfficientNet as feature extractor."""
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
    
    efficient_net = hub.KerasLayer(
        "/kaggle/input/efficientnet-v2/tensorflow2/imagenet21k-b3-feature-vector/1",
        trainable=False
    )
    
    x = efficient_net(inputs)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)  # Probability of fraud
    
    model = keras.Model(inputs, outputs)
    return model

def load_model_weights(model, ckpt_path):
    """Loads the trained weights into the model."""
    model.load_weights(ckpt_path)
    return model

def preprocess_image(img_path):
    """Loads and preprocesses the image to match model input."""
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

def predict_fraud(img_path):
    """Predicts the fraud probability for a given image."""
    model = build_model()
    model = load_model_weights(model, ckpt_path)
    
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]  # Get probability score
    
    print(f"Fraud Probability: {prediction:.4f}")
    return prediction


# Define all possible categories for each categorical column with exact values from training data
CATEGORY_MAPPINGS = {
    'Month': ['Dec   ', 'Jan   ', 'Oct   ', 'Jun   ', 'Feb   ', 'Nov   ', 'Apr   ', 'Mar   ', 'Aug   ', 'Jul   ', 'May   ', 'Sep   '],
    'DayOfWeek': ['Wednesday ', 'Friday    ', 'Saturday  ', 'Monday    ', 'Tuesday   ', 'Sunday    ', 'Thursday  '],
    'Make': ['Honda     ', 'Toyota    ', 'Ford       ', 'Mazda     ', 'Chevrolet ', 'Pontiac   ', 'Accura    ', 'Dodge     ', 'Mercury   ', 'Jaguar    ', 'Nisson    ', 'VW        ', 'Saab      ', 'Saturn    ', 'Porche    ', 'BMW       ', 'Mecedes   ', 'Ferrari   ', 'Lexus     '],
    'AccidentArea': ['Urban        ', 'Rural        '],
    'DayOfWeekClaimed': ['Tuesday          ', 'Monday           ', 'Thursday         ', 'Friday           ', 'Wednesday        ', 'Saturday         ', 'Sunday           ', '0                '],
    'MonthClaimed': ['Jan          ', 'Nov          ', 'Jul          ', 'Feb          ', 'Mar          ', 'Dec          ', 'Apr          ', 'Aug          ', 'May          ', 'Jun          ', 'Sep          ', 'Oct          ', '0            '],
    'Sex': ['Female ', 'Male   '],
    'MaritalStatus': ['Single        ', 'Married       ', 'Widow         ', 'Divorced      '],
    'Fault': ['Policy Holder ', 'Third Party   '],
    'PolicyType': ['Sport - Liability    ', 'Sport - Collision    ', 'Sedan - Liability    ', 'Utility - All Perils ', 'Sedan - All Perils   ', 'Sedan - Collision    ', 'Utility - Collision  ', 'Utility - Liability  ', 'Sport - All Perils   '],
    'VehicleCategory': ['Sport           ', 'Utility         ', 'Sedan           '],
    'VehiclePrice': ['more than 69,000 ', '20,000 to 29,000 ', '30,000 to 39,000 ', 'less than 20,000 ', '40,000 to 59,000 ', '60,000 to 69,000 '],
    'PoliceReportFiled': ['No                ', 'Yes               '],
    'WitnessPresent': ['No             ', 'Yes            '],
    'AgentType': ['External  ', 'Internal  '],
    'AddressChange-Claim': ['1 year              ', 'no change           ', '4 to 8 years        ', '2 to 3 years        ', 'under 6 months      '],
    'BasePolicy': ['Liability  ', 'Collision  ', 'All Perils ']
}

def range_to_average(value):
    if isinstance(value, str) and 'to' in value:
        # Handle commas in numbers and extract only the numeric parts
        parts = value.replace(',', '').strip().split(' to ')
        if len(parts) == 2:
            try:
                start = int(''.join(filter(str.isdigit, parts[0])))
                end = int(''.join(filter(str.isdigit, parts[1])))
                return (start + end) / 2
            except ValueError:
                return value
    elif isinstance(value, str):
        # Handle special cases
        if 'less than' in value:
            try:
                num = int(''.join(filter(str.isdigit, value)))
                return num / 2  # Assume half of the upper limit
            except ValueError:
                return value
        elif 'more than' in value:
            try:
                num = int(''.join(filter(str.isdigit, value)))
                return num * 1.5  # Assume 50% more than the lower limit
            except ValueError:
                return value
        elif value.strip() == 'over 65           ':
            return 70.0
        elif value.strip() == 'new          ':
            return 0.0
        # Try to extract just digits if they exist
        digits = ''.join(filter(str.isdigit, value))
        if digits:
            try:
                return int(digits)
            except ValueError:
                return value
    return value

def standardize_input(value, options):
    """Match input value to closest option with trailing spaces"""
    if isinstance(value, str):
        value_stripped = value.strip()
        for option in options:
            if option.strip() == value_stripped:
                return option
    return value

def encode_categorical_columns(df, category_mappings):
    encoded_dfs = []
    
    # Handle non-categorical columns
    non_cat_cols = [col for col in df.columns if col not in category_mappings]
    if non_cat_cols:
        encoded_dfs.append(df[non_cat_cols])
    
    # Encode each categorical column
    for col, categories in category_mappings.items():
        if col in df.columns:
            # Standardize input value to match training data format
            if df[col].iloc[0] in categories:
                standardized_value = df[col].iloc[0]
            else:
                standardized_value = standardize_input(df[col].iloc[0], categories)
            
            if standardized_value not in categories:
                standardized_value = categories[0]  # Default to first category if no match
            
            # Create a new DataFrame for this column with one-hot encoding
            dummies = pd.DataFrame(0, index=df.index, columns=[f"{col}_{cat}" for cat in categories[1:]])
            
            # Set the corresponding category to 1
            if standardized_value != categories[0]:
                col_name = f"{col}_{standardized_value}"
                if col_name in dummies.columns:
                    dummies[col_name] = 1
            
            encoded_dfs.append(dummies)
    
    # Concatenate all encoded features
    return pd.concat(encoded_dfs, axis=1)

def predict_fraud_probability(
    age_of_vehicle=None,
    age_of_policy_holder=None,
    deductible=None,
    driver_rating=None,
    month="Jan",
    week_of_month=1,
    day_of_week="Monday",
    make="Honda",
    accident_area="Urban",
    day_of_week_claimed="Monday",
    month_claimed="Jan",
    week_of_month_claimed=1,
    sex="Male",
    marital_status="Single",
    fault="Third Party",
    policy_type="Sport - Liability",
    vehicle_category="Sport",
    vehicle_price="20,000 to 29,000",
    days_policy_accident="none",
    days_policy_claim="none",
    past_number_of_claims="none",
    police_report_filed="No",
    witness_present="No",
    agent_type="External",
    number_of_suppliments="none",
    address_change_claim="no change",
    number_of_cars="1 vehicle",
    year=2025,
    base_policy="Liability",
    model_path="fraud_detection_model.pkl"
):
    try:
        # Suppress specific warning about model version
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator")
            
            # Check if model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Create a single row DataFrame with the input data
            input_data = {
                'Month': month,
                'WeekOfMonth': week_of_month,
                'DayOfWeek': day_of_week,
                'Make': make,
                'AccidentArea': accident_area,
                'DayOfWeekClaimed': day_of_week_claimed,
                'MonthClaimed': month_claimed,
                'WeekOfMonthClaimed': week_of_month_claimed,
                'Sex': sex,
                'MaritalStatus': marital_status,
                'Fault': fault,
                'PolicyType': policy_type,
                'VehicleCategory': vehicle_category,
                'VehiclePrice': vehicle_price,
                'Deductible': deductible if deductible is not None else 0,
                'DriverRating': driver_rating if driver_rating is not None else 0,
                'Days:Policy-Accident': days_policy_accident,
                'Days:Policy-Claim': days_policy_claim,
                'PastNumberOfClaims': past_number_of_claims,
                'AgeOfVehicle': age_of_vehicle if age_of_vehicle is not None else 0,
                'AgeOfPolicyHolder': age_of_policy_holder if age_of_policy_holder is not None else 0,
                'PoliceReportFiled': police_report_filed,
                'WitnessPresent': witness_present,
                'AgentType': agent_type,
                'NumberOfSuppliments': number_of_suppliments,
                'AddressChange-Claim': address_change_claim,
                'NumberOfCars': number_of_cars,
                'Year': year,
                'BasePolicy': base_policy
            }
            
            df = pd.DataFrame([input_data])
            
            # Process range values for specific columns
            range_columns = ['AgeOfVehicle', 'AgeOfPolicyHolder', 'NumberOfCars']
            for col in range_columns:
                if col in df.columns:
                    df[col] = df[col].apply(range_to_average)
            
            # Handle other range values in other columns
            for col in df.columns:
                if col not in range_columns and col not in CATEGORY_MAPPINGS:
                    df[col] = df[col].apply(range_to_average)
            
            # Use the custom encoding function for categorical columns
            df_encoded = encode_categorical_columns(df, CATEGORY_MAPPINGS)
            
            try:
                # Load the model
                model = joblib.load(model_path)
                
                # Ensure all required features are present
                missing_cols = set(model.feature_names_in_) - set(df_encoded.columns)
                for col in missing_cols:
                    df_encoded[col] = 0
                
                # Remove extra columns that weren't in training
                extra_cols = set(df_encoded.columns) - set(model.feature_names_in_)
                if extra_cols:
                    df_encoded = df_encoded.drop(columns=list(extra_cols))
                
                # Ensure columns are in the same order as during training
                df_encoded = df_encoded[model.feature_names_in_]
                
                # Make prediction
                fraud_probability = model.predict_proba(df_encoded)[0][1]
                
                return fraud_probability
                
            except (AttributeError, ValueError) as e:
                print("Warning: Model version mismatch detected. Please retrain the model using the same scikit-learn version.")
                print("Original error:", str(e))
                return None
                
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

