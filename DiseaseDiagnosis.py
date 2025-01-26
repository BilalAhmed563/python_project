import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Prepare a larger dataset
data = {
    'Fever': [98.1, 101.3, 99.5, 100.4, 102.2, 97.9, 98.6, 100.1, 99.2, 101.7,
              100.5, 98.2, 97.8, 99.9, 101.8, 100.0, 102.0, 98.7, 99.1, 97.5],
    'Cough': [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    'Fatigue': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    'Headache': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    'Disease': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0]  # 1: Disease present, 0: Disease absent
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Fever', 'Cough', 'Fatigue', 'Headache']]
y = df['Disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Step 3: Enhance Tkinter GUI with patient-doctor interaction and colorful UI
def predict_disease():
    try:
        # Get user inputs
        fever = float(entry_fever.get())
        cough = int(entry_cough.get())
        fatigue = int(entry_fatigue.get())
        headache = int(entry_headache.get())

        # Make prediction (with feature names)
        features = pd.DataFrame([[fever, cough, fatigue, headache]], columns=['Fever', 'Cough', 'Fatigue', 'Headache'])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100

        # Display result
        if prediction == 1:
            result = f"Disease Present\nConfidence: {probability:.2f}%"
        else:
            result = f"No Disease\nConfidence: {100 - probability:.2f}%"

        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid inputs.")


def show_dataset_info():
    info = f"Dataset Info:\n\nTotal Records: {len(df)}\nTraining Records: {len(X_train)}\nTesting Records: {len(X_test)}\nModel Accuracy: {accuracy * 100:.2f}%"
    messagebox.showinfo("Dataset Info", info)

def show_about():
    about_text = (
        "Disease Diagnosis Predictor\n\n"
        "This application uses a logistic regression model to predict the presence of a disease "
        "based on symptoms such as Fever, Cough, Fatigue, and Headache.\n\n"
        "Developed by: [Your Name]\nVersion: 1.0"
    )
    messagebox.showinfo("About", about_text)

# Initialize Tkinter window
# Initialize Tkinter window
app = tk.Tk()
app.title("Disease Diagnosis Predictor")
app.geometry("500x400")  # Optional: Set a fixed window size
app.configure(bg="#d4f1f4")  # Light blue background for the app

# Create colorful header
header_label = tk.Label(
    app,
    text="Disease Diagnosis System",
    font=("Helvetica", 16, "bold"),
    bg="#4682b4",  # Steel blue color
    fg="white",
    pady=10
)
header_label.grid(row=0, column=0, columnspan=2, sticky="ew")

# Create a section for instructions
label_instructions = tk.Label(
    app,
    text="Doctor: Please answer the following questions:",
    font=("Helvetica", 12),
    bg="#d4f1f4",  # Light blue background
    fg="#333"      # Darker text color
)
label_instructions.grid(row=1, column=0, columnspan=2, pady=10)

# Style input fields and labels
label_fever = tk.Label(
    app,
    text="Do you have a fever? (e.g., 98.6):",
    font=("Helvetica", 10),
    bg="#d4f1f4",
    fg="#333"
)
label_fever.grid(row=2, column=0, padx=10, pady=10, sticky="e")
entry_fever = tk.Entry(app, bg="#ffffff", fg="#333")
entry_fever.grid(row=2, column=1, padx=10, pady=10)

label_cough = tk.Label(
    app,
    text="Do you have a cough? (1: Yes, 0: No):",
    font=("Helvetica", 10),
    bg="#d4f1f4",
    fg="#333"
)
label_cough.grid(row=3, column=0, padx=10, pady=10, sticky="e")
entry_cough = tk.Entry(app, bg="#ffffff", fg="#333")
entry_cough.grid(row=3, column=1, padx=10, pady=10)

label_fatigue = tk.Label(
    app,
    text="Do you feel fatigued? (1: Yes, 0: No):",
    font=("Helvetica", 10),
    bg="#d4f1f4",
    fg="#333"
)
label_fatigue.grid(row=4, column=0, padx=10, pady=10, sticky="e")
entry_fatigue = tk.Entry(app, bg="#ffffff", fg="#333")
entry_fatigue.grid(row=4, column=1, padx=10, pady=10)

label_headache = tk.Label(
    app,
    text="Do you have a headache? (1: Yes, 0: No):",
    font=("Helvetica", 10),
    bg="#d4f1f4",
    fg="#333"
)
label_headache.grid(row=5, column=0, padx=10, pady=10, sticky="e")
entry_headache = tk.Entry(app, bg="#ffffff", fg="#333")
entry_headache.grid(row=5, column=1, padx=10, pady=10)

# Add colorful buttons
predict_button = tk.Button(
    app,
    text="Predict",
    font=("Helvetica", 10, "bold"),
    bg="#32cd32",  # Lime green for predict button
    fg="white",
    command=predict_disease
)
predict_button.grid(row=6, column=0, columnspan=2, pady=10)

dataset_info_button = tk.Button(
    app,
    text="Dataset Info",
    font=("Helvetica", 10, "bold"),
    bg="#ffa500",  # Orange for dataset info
    fg="white",
    command=show_dataset_info
)
dataset_info_button.grid(row=7, column=0, columnspan=2, pady=10)

about_button = tk.Button(
    app,
    text="About",
    font=("Helvetica", 10, "bold"),
    bg="#1e90ff",  # Dodger blue for about button
    fg="white",
    command=show_about
)
about_button.grid(row=8, column=0, columnspan=2, pady=10)



# Run the Tkinter loop
app.mainloop()

