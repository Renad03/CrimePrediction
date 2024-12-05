import customtkinter as ctk
import numpy as np
import joblib
from tkinter import messagebox
import webview

# Load saved models and preprocessors
log_reg = joblib.load('logistic_regression_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('minmax_scaler.pkl')
le_location_description = joblib.load('label_encoder_Location_Description.pkl')
le_primary_type = joblib.load('label_encoder_Primary_Type.pkl')
le_description = joblib.load('label_encoder_Description.pkl')
le_block = joblib.load('label_encoder_Block.pkl')
le_domestic = joblib.load('label_encoder_domestic.pkl')

# Function to make predictions
def make_prediction():
    try:
        # Collect user inputs
        year = int(entry_year.get())
        month = int(entry_month.get())
        day = int(entry_day.get())
        day_of_week = int(entry_day_of_week.get())
        latitude = float(entry_latitude.get())
        longitude = float(entry_longitude.get())
        domestic_value = domestic_var.get()  # 1 for True, 0 for False
        primary_type = entry_primary_type.get()
        location_description = entry_location_description.get()
        block = entry_block.get()
        description = entry_description.get()

        # Encode categorical inputs dynamically
        try:
            encoded_primary_type = le_primary_type.transform([primary_type])[0]
        except ValueError:
            encoded_primary_type = len(le_primary_type.classes_)
            le_primary_type.classes_ = np.append(le_primary_type.classes_, primary_type)

        try:
            encoded_location_description = le_location_description.transform([location_description])[0]
        except ValueError:
            encoded_location_description = len(le_location_description.classes_)
            le_location_description.classes_ = np.append(le_location_description.classes_, location_description)

        try:
            encoded_block = le_block.transform([block])[0]
        except ValueError:
            encoded_block = len(le_block.classes_)
            le_block.classes_ = np.append(le_block.classes_, block)

        try:
            encoded_description = le_description.transform([description])[0]
        except ValueError:
            encoded_description = len(le_description.classes_)
            le_description.classes_ = np.append(le_description.classes_, description)

        input_data = np.array([[year, month, day, day_of_week, latitude, longitude,
                                domestic_value, encoded_primary_type, encoded_location_description,
                                encoded_block, encoded_description]])

        # Scale all features (numerical + categorical)
        input_scaled = scaler.transform(input_data)

        # Logistic Regression predictions
        log_reg_prediction = log_reg.predict(input_scaled)[0]
        log_reg_probability = log_reg.predict_proba(input_scaled)[0][1]

        # Decision Tree predictions
        dt_prediction = dt_model.predict(input_scaled)[0]
        dt_probability = dt_model.predict_proba(input_scaled)[0][1]

        # Display results
        result_text = (
            f"Logistic Regression:\n"
            f"  Predicted Arrest: {'Yes' if log_reg_prediction == 1 else 'No'}\n"
            f"  Probability: {log_reg_probability:.2f}\n\n"
            f"Decision Tree:\n"
            f"  Predicted Arrest: {'Yes' if dt_prediction == 1 else 'No'}\n"
            f"  Probability: {dt_probability:.2f}"
        )
        result_label.configure(text=result_text)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to create the CustomTkinter GUI
def create_gui():
    global entry_year, entry_month, entry_day, entry_day_of_week, entry_latitude, entry_longitude
    global domestic_var, entry_primary_type, entry_location_description, entry_block, entry_description
    global result_label

    # Initialize CTk window
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    root.title("Crime Prediction System")
    root.geometry("800x800")  # Set window size

    # Input fields
    ctk.CTkLabel(root, text="Year:").pack()
    entry_year = ctk.CTkEntry(root)
    entry_year.pack()

    ctk.CTkLabel(root, text="Month:").pack()
    entry_month = ctk.CTkEntry(root)
    entry_month.pack()

    ctk.CTkLabel(root, text="Day:").pack()
    entry_day = ctk.CTkEntry(root)
    entry_day.pack()

    ctk.CTkLabel(root, text="Day of Week (0=Monday, 6=Sunday):").pack()
    entry_day_of_week = ctk.CTkEntry(root)
    entry_day_of_week.pack()

    ctk.CTkLabel(root, text="Latitude:").pack()
    entry_latitude = ctk.CTkEntry(root)
    entry_latitude.pack()

    ctk.CTkLabel(root, text="Longitude:").pack()
    entry_longitude = ctk.CTkEntry(root)
    entry_longitude.pack()

    # Radio buttons for Domestic
    ctk.CTkLabel(root, text="Domestic:").pack()
    domestic_var = ctk.IntVar()
    ctk.CTkRadioButton(root, text="Yes", variable=domestic_var, value=1).pack()
    ctk.CTkRadioButton(root, text="No", variable=domestic_var, value=0).pack()

    # Input fields for categorical features
    ctk.CTkLabel(root, text="Primary Type:").pack()
    entry_primary_type = ctk.CTkEntry(root)
    entry_primary_type.pack()

    ctk.CTkLabel(root, text="Location Description:").pack()
    entry_location_description = ctk.CTkEntry(root)
    entry_location_description.pack()

    ctk.CTkLabel(root, text="Block:").pack()
    entry_block = ctk.CTkEntry(root)
    entry_block.pack()

    ctk.CTkLabel(root, text="Description:").pack()
    entry_description = ctk.CTkEntry(root)
    entry_description.pack()

    # Predict button
    predict_button = ctk.CTkButton(root, text="Predict", command=make_prediction)
    predict_button.pack()

    # Result label
    result_label = ctk.CTkLabel(root, text="Enter values and press Predict.")
    result_label.pack()

    return root

# Function to start the PyWebView application
def start_webview():
    # Create the Tkinter GUI first
    root = create_gui()

    # Use PyWebView to render it in a webview
    webview.create_window('Crime Prediction System', root)
    webview.start()

if __name__ == "__main__":
    start_webview()
