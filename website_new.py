import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import gdown
import tempfile
import shutil
import requests
from tqdm import tqdm

# Set page config
st.set_page_config(
    page_title="Microbiome Symptom Predictor",
    page_icon="🦠",
    layout="wide"
)

class MicrobiomeNet(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, output_size=2):
        super(MicrobiomeNet, self).__init__()
        
        # Feature attention network
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Abundance processing network
        self.abundance_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Interaction processing network
        self.interaction_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # Apply feature attention
        attention = torch.sigmoid(self.feature_attention(x))
        x_attended = x * attention
        
        # Process through parallel networks
        abundance_features = self.abundance_network(x_attended)
        interaction_features = self.interaction_network(x)
        
        # Combine features
        combined = torch.cat([abundance_features, interaction_features], dim=1)
        
        # Final processing
        output = self.final_layers(combined)
        return output

def download_models_from_gdrive(file_id="1--s3u-BiIeoluB_ji97YE5cH13Se3dum"):
    """Download zipped models from Google Drive using gdown"""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "models.zip")
    st.info("Downloading models from Google Drive...")
    
    try:
        # Construct URL with export and confirm parameters
        url = f"https://drive.google.com/u/0/uc?id={file_id}&export=download&confirm=t"
        
        # Use gdown with the specific URL format
        st.write("Starting download...")
        output = gdown.download(
            url,
            zip_path,
            quiet=False,
            fuzzy=True
        )
        
        if output is None:
            raise Exception("Download failed - gdown returned None")
            
        # Debug: Check file size and signature
        if os.path.exists(zip_path):
            actual_size = os.path.getsize(zip_path)
            st.write(f"Downloaded file size: {actual_size / (1024*1024):.2f} MB")
            
            # Check file signature
            with open(zip_path, 'rb') as f:
                first_bytes = f.read(4).hex()
                st.write(f"File signature: {first_bytes}")
                # ZIP file should start with PK (0x504B)
                if not first_bytes.startswith('504b'):
                    raise Exception("Downloaded file is not a valid ZIP (incorrect file signature)")
        else:
            raise Exception("Download completed but file not found at expected location")
        
        st.write("Extracting files...")
        # Extract the zip file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            st.write("Files extracted successfully")
            
        # List contents of temp_dir for debugging
        st.write("Contents of temp directory:", os.listdir(temp_dir))
            
        # Remove the zip file
        os.remove(zip_path)
        
        return temp_dir
        
    except Exception as e:
        st.error(f"Error downloading models: {str(e)}")
        if os.path.exists(zip_path):
            # If the file exists but isn't a valid ZIP, save it for inspection
            debug_path = "debug_download.bin"
            shutil.copy(zip_path, debug_path)
            st.error(f"Invalid file saved to {debug_path} for debugging")
        st.error(f"Temp directory contents: {os.listdir(temp_dir) if os.path.exists(temp_dir) else 'directory not found'}")
        shutil.rmtree(temp_dir)
        return None

def load_saved_models():
    """Load all saved models from Google Drive"""
    models = {}
    scalers = {}
    pcas = {}
    
    # Download models to temporary directory
    temp_dir = download_models_from_gdrive()
    if not temp_dir:
        raise Exception("Failed to download models from Google Drive")
    
    try:
        # Load models from temporary directory
        models_dir = os.path.join(temp_dir, "saved_models")
        
        for filename in os.listdir(models_dir):
            if filename.endswith("_model.pth"):
                # Extract symptom name and handle special characters
                symptom = filename.replace("_model.pth", "")
                model_path = os.path.join(models_dir, filename)
                scaler_path = os.path.join(models_dir, f"{symptom}_scaler.joblib")
                pca_path = os.path.join(models_dir, f"{symptom}_pca.joblib")
                
                # Initialize and load model
                model = MicrobiomeNet(input_size=1024, hidden_size=128, output_size=2)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                
                # Load scaler and PCA
                scaler = joblib.load(scaler_path)
                pca = joblib.load(pca_path)
                
                models[symptom] = model
                scalers[symptom] = scaler
                pcas[symptom] = pca
        
        st.write(f"Loaded {len(models)} models successfully")
        return models, scalers, pcas
        
    except Exception as e:
        st.error(f"Error in load_saved_models: {str(e)}")
        raise
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def process_species_data(file):
    """Process the uploaded species TSV file"""
    df = pd.read_csv(file, sep='\t')
    
    # Extract abundance and species columns
    abundance_data = df[['%_Abundance', 'Species_Name']]
    
    # Pivot the data to get species as columns
    pivoted_data = abundance_data.pivot_table(
        index=None, 
        values='%_Abundance', 
        columns='Species_Name', 
        aggfunc='sum'
    ).fillna(0)
    
    return pivoted_data

def predict_symptoms(data, models, scalers, pcas):
    """Make predictions for each symptom"""
    predictions = {}
    
    for symptom, model in models.items():
        try:
            # Get the feature names from the scaler
            scaler_features = scalers[symptom].get_feature_names_out()
            
            # Create a DataFrame with zeros for all scaler features
            prediction_data = pd.DataFrame(0, index=[0], columns=scaler_features)
            
            # Fill in the available species data
            common_species = data.columns.intersection(scaler_features)
            prediction_data[common_species] = data[common_species]
            
            # Scale the data
            scaled_data = scalers[symptom].transform(prediction_data)
            
            # Apply PCA transformation
            pca_data = pcas[symptom].transform(scaled_data)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(pca_data)
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.sigmoid(output).numpy()
            
            predictions[symptom] = prediction[0][0]
            
        except Exception as e:
            st.error(f"Error predicting {symptom}: {str(e)}")
            continue
    
    return predictions

def get_friendly_symptom_name(symptom):
    """Convert the long symptom names to friendly display names"""
    # Dictionary mapping original names to friendly names
    name_mapping = {
        "How_much_does_these_symptoms_bother_your_daily_life_from_1-10?__(Please_respond_for_all_symptoms)_Bloating": "Bloating Severity",
        "How_much_does_these_symptoms_bother_your_daily_life_from_1-10?__(Please_respond_for_all_symptoms)_Acidity_Burning": "Acidity Severity",
        "How_much_does_these_symptoms_bother_your_daily_life_from_1-10?__(Please_respond_for_all_symptoms)_Constipation": "Constipation Severity",
        "How_much_does_these_symptoms_bother_your_daily_life_from_1-10?__(Please_respond_for_all_symptoms)_Loose_Motion_Diarrhea": "Diarrhea Severity",
        "How_much_does_these_symptoms_bother_your_daily_life_from_1-10?__(Please_respond_for_all_symptoms)_Flatulence_Gas_Fart": "Gas Severity",
        "How_much_does_these_symptoms_bother_your_daily_life_from_1-10?__(Please_respond_for_all_symptoms)_Burping": "Burping Severity",
        "How_many_days_in_a_week_do_you_generally_experience_the_following_symptoms?_(Please_respond_for_all_symptoms)_Acidity": "Acidity Frequency",
        "How_many_days_in_a_week_do_you_generally_experience_the_following_symptoms?_(Please_respond_for_all_symptoms)_Bloating": "Bloating Frequency",
        "How_many_days_in_a_week_do_you_generally_experience_the_following_symptoms?_(Please_respond_for_all_symptoms)_Burping": "Burping Frequency",
        "How_many_days_in_a_week_do_you_generally_experience_the_following_symptoms?_(Please_respond_for_all_symptoms)_Constipation": "Constipation Frequency",
        "How_many_days_in_a_week_do_you_generally_experience_the_following_symptoms?_(Please_respond_for_all_symptoms)_Flatulence_Gas_Fart": "Gas Frequency"
    }
    return name_mapping.get(symptom, symptom)

def main():
    st.title("🦠 Microbiome Symptom Predictor")
    
    # Load saved models
    try:
        models, scalers, pcas = load_saved_models()
        st.success("Models loaded successfully!")
        
        # Display some model info
        sample_scaler = next(iter(scalers.values()))
        n_features = len(sample_scaler.get_feature_names_out())
        st.info(f"Models expect {n_features} species features and will use PCA to reduce to 1024 dimensions.")
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # File upload
    st.header("Upload Species Data")
    uploaded_file = st.file_uploader(
        "Upload your species abundance TSV file", 
        type=['tsv'],
        help="Upload a TSV file containing species abundance data"
    )

    if uploaded_file is not None:
        try:
            # Process the uploaded file
            species_data = process_species_data(uploaded_file)
            
            # Show some data info
            st.info(f"Processed {len(species_data.columns)} species from your data.")
            
            # Make predictions
            predictions = predict_symptoms(species_data, models, scalers, pcas)
            
            if predictions:
                # Display results
                st.header("Prediction Results")
                
                # Create two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Scores")
                    # Create a DataFrame for the predictions with friendly names
                    pred_df = pd.DataFrame({
                        'Symptom': [get_friendly_symptom_name(k) for k in predictions.keys()],
                        'Probability': list(predictions.values())
                    })
                    
                    # Display as table
                    st.dataframe(pred_df.style.format({'Probability': '{:.2%}'}))
                
                with col2:
                    st.subheader("Visualization")
                    # Create bar plot with friendly names
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[get_friendly_symptom_name(k) for k in predictions.keys()],
                            y=list(predictions.values()),
                            text=[f"{v:.1%}" for v in predictions.values()],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Symptom Prediction Probabilities",
                        xaxis_title="Symptoms",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        template="plotly_white",
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # Rotate x-axis labels for better readability
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        margin=dict(b=100)  # Add bottom margin for rotated labels
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Error details:", str(e))
            st.write("Please ensure your TSV file:")
            st.write("1. Contains '%_Abundance' and 'Species_Name' columns")
            st.write("2. Is properly formatted")
            st.write("3. Contains species that match the training data")

    # Add information about the expected format
    with st.expander("ℹ️ Input Format Information"):
        st.write("""
        Your TSV file should contain the following columns:
        - %_Abundance: Numerical values representing species abundance
        - Species_Name: Names of the species
        - Tax_ID: Taxonomy IDs (optional)
        - Taxonomy: Full taxonomy information (optional)
        
        Only the abundance and species name columns will be used for prediction.
        """)

if __name__ == "__main__":
    main()
