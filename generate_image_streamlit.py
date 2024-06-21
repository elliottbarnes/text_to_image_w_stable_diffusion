import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

st.title("Text-to-Image Generator")

prompt = st.text_input("Enter a text prompt:")

if st.button("Generate"):
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    
    with torch.no_grad():
        output = pipe(prompt)
        image = output.images[0]
        
    # Ensure the image is in the correct format
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    st.image(image, caption="Generated Image")
    image.save("generated_image.png")

