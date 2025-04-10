import streamlit as st
import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image, ImageOps, ImageEnhance
from streamlit_drawable_canvas import st_canvas
import csv
from datetime import datetime
import psycopg2

# Load the trained PyTorch model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model and load weights
model = SimpleNN()
model.load_state_dict(torch.load("mnist_pytorch_model.pth"))
model.eval()

# Define preprocessing for the input image
transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

# Streamlit app
st.title("MNIST Digit Classifier")
st.write("Draw a digit (0-9) on the canvas and enter the true label to submit.")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="#FFFFFF",  # Background color
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Input field for the true label
true_label = st.text_input("Enter the true label (0-9):")

# Submit button
if st.button("Submit"):
    if canvas_result is not None and canvas_result.image_data is not None and true_label:
        # Save the canvas image
        image = Image.fromarray((canvas_result.image_data).astype("uint8"))
        image = image.convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Save the processed image for debugging
        image.save("digit_drawing.png")

        # Convert to tensor and normalize
        image_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            digit = predicted.item()

        st.image(image, caption="Processed Drawing")

        # Save feedback to a CSV file, including confidence
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        confidence = torch.softmax(output, dim=1).numpy()[0][digit]
        try:
            with open("feedback_table.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, digit, confidence, true_label])
            st.write("Feedback saved to table.")
        except Exception as e:
            st.write(f"Error writing to file: {e}")

        # Display results
        st.write(f"Predicted Digit: {digit}")
        st.write(f"True Label: {true_label}")
        st.write(f"Confidence: {confidence:.2f}")
        st.write(output)
        
    else:
        st.write("Please draw a digit and enter the true label before submitting.")

try:
        # Read the feedback table
        with open("feedback_table.csv", mode="r") as file:
            reader = csv.reader(file)
            history = list(reader)

        # Display the history as a table
        if history:
            st.write("### Feedback History")
            st.table(history)
        else:
            st.write("No feedback history available.")
except FileNotFoundError:
    st.write("No feedback history available.")

# Function to log feedback to PostgreSQL
def log_to_postgresql(timestamp, predicted_label, true_label, confidence):
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            dbname="feedback_db",
            user="username",
            password="password",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                predicted_label INT,
                true_label INT,
                confidence FLOAT
            )
        ''')

        # Insert feedback into the table
        cursor.execute('''
            INSERT INTO feedback (timestamp, predicted_label, true_label, confidence)
            VALUES (%s, %s, %s, %s)
        ''', (timestamp, predicted_label, true_label, confidence))

        # Commit and close connection
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.write(f"Error logging to PostgreSQL: {e}")

        # Save feedback to PostgreSQL
        log_to_postgresql(timestamp, digit, true_label, confidence)



