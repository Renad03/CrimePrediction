import streamlit as st
import numpy as np
import joblib
import requests
import streamlit_lottie as st_lottie
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
st.set_page_config(
    page_title="crime predection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
log_reg = joblib.load("logistic_reg_model")
dt_model = joblib.load("decision_tree_model")
rf_model = joblib.load("random_forest_model.joblib")
knn_model = joblib.load("knn_model")
svm_model = joblib.load("svm_model")
encoding_map = pd.read_csv("encoding_map_primary.csv")
encoding_map_block=pd.read_csv("encoding_map_block.csv")
encoding_map_IUCR=pd.read_csv("encoding_map_IUCR.csv")
encoding_map_disc=pd.read_csv("encoding_map_disc.csv")
encoding_map_discription=pd.read_csv("encoding_map_discription.csv")
encoding_map_domestic=pd.read_csv("encoding_map_domestic.csv")
log_metrics=pd.read_csv("log_metrics")
Knn_metrics=pd.read_csv("knn_metrics")
svm_metrics=pd.read_csv("svm_metrics")
dt_metrics=pd.read_csv("dt_metrics")
rf_metrics=pd.read_csv("rfc_metrics.csv")

  
def get_encoded_value(encoding_map, input_value):
    # Check if the entered value exists in the encoding map and fetch its encoded value
    matching_row = encoding_map[encoding_map['Original Value'] == input_value]
    # Return the encoded value if found, else return the default encoded value
    return matching_row['Encoded Value'].values[0] if not matching_row.empty else 0
    
with st.container():
    col1, col2 = st.columns([5, 5], gap="large")

    with col1:
        model_option = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "KNN", "SVM","Random Forest"])
        if model_option=="Random Forest":
            st.subheader("Input")
            latitude = st.number_input("Latitude:", value=0.0)
            longitude = st.number_input("Longitude:", value=0.0)
            
            primary_type_options = encoding_map['Original Value'].tolist()
            primary_type = st.selectbox("Primary Type:", primary_type_options) 

            location_description_options = encoding_map_disc['Original Value'].to_list()
            location_description=st.selectbox("Location Description:",location_description_options)

            description_options = encoding_map_discription['Original Value'].to_list()
            description=st.selectbox("Description:",description_options)

            domestic_options=encoding_map_domestic['Original Value'].to_list()
            domestic=st.selectbox("domestic:",domestic_options)

            block_options=encoding_map_block['Original Value'].to_list()
            Block = st.selectbox("Block:",block_options)
            district = st.number_input("district:", value=0.0)
            Iucr_options=encoding_map_IUCR['Original Value'].to_list()
            Iucr = st.selectbox("IUCR:",Iucr_options)
            ward = st.number_input("Ward:")
            community_area = st.number_input("Community Area:")
            beat = st.number_input("Beat:")
            primary_type_encoded = get_encoded_value(encoding_map, primary_type.strip()) 
            block = get_encoded_value(encoding_map_block, Block) 
            IUCR_encoded = get_encoded_value(encoding_map_IUCR, Iucr) 
            location_description_encoded = get_encoded_value(encoding_map_disc, location_description) 
            description_encoded = get_encoded_value(encoding_map_discription, description) 
            domestic_encoded=get_encoded_value(encoding_map_domestic, domestic) 
        else:
            st.subheader("Input")
            latitude = st.number_input("Latitude:", value=0.0)
            longitude = st.number_input("Longitude:", value=0.0)
            
            primary_type_options = encoding_map['Original Value'].tolist()
            primary_type = st.selectbox("Primary Type:", primary_type_options) 

            location_description_options = encoding_map_disc['Original Value'].to_list()
            location_description=st.selectbox("Location Description:",location_description_options)

            description_options = encoding_map_discription['Original Value'].to_list()
            description=st.selectbox("Description:",description_options)

            block_options=encoding_map_block['Original Value'].to_list()
            Block = st.selectbox("Block:",block_options)
            Iucr_options=encoding_map_IUCR['Original Value'].to_list()
            Iucr = st.selectbox("IUCR:",Iucr_options)
            ward = st.number_input("Ward:")
            community_area = st.number_input("Community Area:")
            beat = st.number_input("Beat:")
            
            primary_type_encoded = get_encoded_value(encoding_map, primary_type.strip()) 
            block = get_encoded_value(encoding_map_block, Block) 
            IUCR_encoded = get_encoded_value(encoding_map_IUCR, Iucr) 
            location_description_encoded = get_encoded_value(encoding_map_disc, location_description) 
            description_encoded = get_encoded_value(encoding_map_discription, description)
            domestic_options=encoding_map_domestic['Original Value'].to_list()
            domestic=st.selectbox("domestic:",domestic_options)
            domestic_encoded=get_encoded_value(encoding_map_domestic, domestic) 
            district = st.number_input("district:", value=0.0)


    st.write("               ")  
   # Feature Importance Visualization
    with col2:
        st.subheader("Feature Importance")
        feature_importance_option = st.selectbox("Select Option", ["Show Feature Importance", "Hide Feature Importance"])
        if feature_importance_option == "Show Feature Importance":
            try:
                # Assuming the pre-trained Random Forest model (rf_model) is loaded
                # Extract feature importances from the Random Forest Classifier step in the pipeline
                feature_importances = pd.Series(
                    rf_model.named_steps["randomforestclassifier"].feature_importances_,  # Access RF step directly
                    index=["block", "IUCR", "primary_type", "description", "location_description", "domestic", "latitude", "longitude", "beat", "ward", "community_area", "district"]  # List of feature names
                ).sort_values(ascending=False)

                # Plotting feature importance
                plt.figure(figsize=(6, 4))
                sns.barplot(x=feature_importances.values, y=feature_importances.index, orient='h')

                plt.xlabel('Feature Importance Score')
                plt.ylabel('Features')
                plt.title('Visualizing Important Features')
                st.pyplot(plt)

            except Exception as e:
                st.error(f"Error generating plot: {e}")


# Handle the encoding of each user input and store the results




# Predict button
if st.button("Predict"):
    try:  
        

        models = {
            "Logistic Regression": log_reg,
            "Decision Tree": dt_model,
            "KNN": knn_model,
            "SVM": svm_model,
            "Random Forest":rf_model
        }
        selected_model = models[model_option]

        input_data = pd.DataFrame({
        'Block': [block], 
        'IUCR': [IUCR_encoded], 
        'Primary Type': [primary_type_encoded],
        'Description': [description_encoded], 
        'Location Description': [location_description_encoded], 
        'Domestic': [domestic_encoded],
        'Beat': [beat],
        'District': [district],
        'Ward': [ward],
        'Community Area': [community_area],
        'Latitude': [latitude],
        'Longitude': [longitude]
        
        })
        prediction = selected_model.predict(input_data)[0]
        
        metrics=None
        if model_option=="Logistic Regression":
            metrics=log_metrics
        elif model_option=="Decision Tree":
            metrics=dt_metrics
        elif model_option=="KNN":
            metrics=Knn_metrics
        elif model_option=="SVM":
            metrics=svm_metrics
        elif model_option=="Random Forest":
            metrics=rf_metrics
        else:
            st.error(f"Model {model_option} not recognized.")
            metrics = {} 

        accuracy = metrics["accuracy"].values[0]
        confusion_matrix = metrics["confusion_matrix"].values[0]
        conf_matrix = ast.literal_eval(confusion_matrix)
        classification_report_str = metrics["classification_report"].iloc[0]

        st.subheader("Model Results")
        st.write("Arrested" if prediction == 1 else "Not Arrested")
        st.write(f"**Accuracy**: {accuracy:.4f}")

            
        st.subheader("Classification Report")

        classification_report_lines = classification_report_str.split("\n")

        report_rows = []

        for line in classification_report_lines:
            row = line.split()
            if len(row) >= 5:  # Ensure we only take the first 5 columns
                report_rows.append(row[:5])  # Take only the first 5 elements (Metrics, Precision, Recall, F1-Score, Support)

        classification_report_df = pd.DataFrame(report_rows, columns=["Metrics", "Precision", "Recall", "F1-Score", "Support"])

        st.table(classification_report_df)

        st.subheader("Confusion Matrix")
        conf_matrix_df = pd.DataFrame(conf_matrix, columns=["Predicted: No Crime", "Predicted: Crime"],
                                    index=["Actual: No Crime", "Actual: Crime"])
        st.table(conf_matrix_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
