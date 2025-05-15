from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import requests
from io import StringIO
from flasgger import Swagger

app = Flask(__name__)

# Swagger config
app.config['SWAGGER'] = {
    'title': 'NYC Airbnb Rental Price Prediction API',
    'uiversion': 3
}
swagger = Swagger(app)

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///nyc_listings.db'
db = SQLAlchemy(app)

# Define a database model
class Listing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    price = db.Column(db.Float, nullable=False)
    bedrooms = db.Column(db.Integer, nullable=False)
    bathrooms = db.Column(db.Float, nullable=False)
    accommodates = db.Column(db.Integer, nullable=False)
    neighbourhood = db.Column(db.String(100), nullable=False)
    room_type = db.Column(db.String(50), nullable=False)

# Create the database
with app.app_context():
    db.create_all()

def preprocess_data(df):
    # Clean the price column (no dollar signs in this dataset)
    df['price'] = df['price'].astype(float)
    
    # Select relevant columns and drop rows with missing values
    df = df[['price', 'bedrooms', 'bathrooms', 'accommodates', 
             'neighbourhood_group', 'room_type']].dropna()
    
    # Fill any remaining missing numerical values with median
    for col in ['bedrooms', 'bathrooms', 'accommodates']:
        df[col] = df[col].fillna(df[col].median())
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False)
    categorical_features = ['neighbourhood_group', 'room_type']
    encoded_features = encoder.fit_transform(df[categorical_features])
    
    # Create DataFrames for encoded features
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Concatenate with numerical features
    df = pd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)
    
    return df, encoder

# Global variables for model and encoder
model = None
encoder = None

@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Reload data from the NYC Airbnb dataset, clear the database, load new data, and return summary stats
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    '''
    global model, encoder

    # Step 1: Download NYC Airbnb data
    url = 'https://raw.githubusercontent.com/your-username/airbnb-price-prediction/main/AB_NYC_2019.csv'
    response = requests.get(url)
    
    # Step 2: Load data into pandas
    listings = pd.read_csv(StringIO(response.text))
    
    # Step 3: Clear the database
    db.session.query(Listing).delete()
    
    # Step 4: Process data and insert into database
    listings = listings[['price', 'bedrooms', 'bathrooms', 'accommodates', 
                        'neighbourhood_group', 'room_type']].dropna()
    
    for _, row in listings.iterrows():
        new_listing = Listing(
            price=float(row['price']),
            bedrooms=int(row['bedrooms']),
            bathrooms=float(row['bathrooms']),
            accommodates=int(row['accommodates']),
            neighbourhood=row['neighbourhood_group'],
            room_type=row['room_type']
        )
        db.session.add(new_listing)
    db.session.commit()
    
    # Step 5: Preprocess and train model
    df, encoder = preprocess_data(listings)
    X = df.drop(columns='price')
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    
    # Step 6: Generate summary statistics
    summary = {
        'total_listings': len(listings),
        'average_price': listings['price'].mean(),
        'min_price': listings['price'].min(),
        'max_price': listings['price'].max(),
        'average_bedrooms': listings['bedrooms'].mean(),
        'average_bathrooms': listings['bathrooms'].mean(),
        'neighbourhood_distribution': listings['neighbourhood_group'].value_counts().to_dict(),
        'room_type_distribution': listings['room_type'].value_counts().to_dict()
    }
    
    return jsonify(summary)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the rental price for an NYC Airbnb listing
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            bedrooms:
              type: integer
            bathrooms:
              type: number
            accommodates:
              type: integer
            neighbourhood_group:
              type: string
              enum: [Manhattan, Brooklyn, Queens, Bronx, Staten Island]
            room_type:
              type: string
              enum: [Entire home/apt, Private room, Shared room]
    responses:
      200:
        description: Predicted rental price
    '''
    global model, encoder

    # Check if model is trained
    if model is None or encoder is None:
        return jsonify({"error": "Model not trained. Please call /reload first."}), 400
    
    data = request.json
    
    try:
        # Validate and convert input
        bedrooms = int(data['bedrooms'])
        bathrooms = float(data['bathrooms'])
        accommodates = int(data['accommodates'])
        neighbourhood = data['neighbourhood_group']
        room_type = data['room_type']
        
        # Prepare input for model
        input_categorical = pd.DataFrame([[neighbourhood, room_type]], 
                                       columns=['neighbourhood_group', 'room_type'])
        encoded_features = encoder.transform(input_categorical)
        
        # Combine numerical and encoded features
        input_data = np.concatenate([
            [bedrooms, bathrooms, accommodates],
            encoded_features[0]
        ]).reshape(1, -1)
        
        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        return jsonify({
            "predicted_price": float(predicted_price),
            "message": "Success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
