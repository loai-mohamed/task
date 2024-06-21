import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load a pre-trained model from torchvision
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

# Preprocess the image for the model
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    return img_tensor

# Load labels from PyTorch
@st.cache_data
def load_labels():
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

st.title("Image Component Recognition App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Button to trigger image analysis
    if st.button('Analyze Image'):
        # Load the model
        model = load_model()
        
        # Load the labels
        labels = load_labels()

        # Preprocess the image
        img_tensor = preprocess_image(image)

        # Predict the image
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # Get the top 5 predictions
        _, indices = torch.topk(outputs, 5)
        percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        predictions = [(labels[idx], percentages[idx].item()) for idx in indices[0]]

        # Display the predictions
        st.write("Predicted components in the image:")
        components = [label for label, _ in predictions]
        st.write(components)


streamlit run app.py
