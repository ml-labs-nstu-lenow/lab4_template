import gradio as gr
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Create the Gradio app
demo = gr.Interface(
    fn=model.predict,
    inputs=[
        gr.Textbox(label="Enter the sepal length"),
        gr.Textbox(label="Enter the sepal width"),
        gr.Textbox(label="Enter the petal length"),
        gr.Textbox(label="Enter the petal width"),
    ],
    outputs=gr.Label(label="Predicted class"),
    title="Iris Classification",
    description="This is a simple Iris classification demo using a Random Forest classifier.",
)
demo.launch()
