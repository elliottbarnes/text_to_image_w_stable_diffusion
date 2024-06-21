import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_image(prompt):
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    
    with torch.no_grad():
        image = pipe(prompt).images[0]
        
    return image

if __name__ == "__main__":
    prompt = input("Enter a text prompt: ")
    image = generate_image(prompt)
    if image:
        image.show()
        image.save("generated_image.png")

