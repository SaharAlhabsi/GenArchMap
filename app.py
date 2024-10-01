from flask import Flask, render_template, request, send_file
from transformers import AutoModel, AutoTokenizer
import diffusers
import torch
from PIL import Image
import os

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Load Stable Diffusion XL Base1.0
pipe = diffusers.DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    # torch_dtype=torch.float16,
    # variant="fp16",
    use_safetensors=True
).to(device)

# Optional CPU offloading to save some GPU Memory
pipe.enable_model_cpu_offload()

# Loading Trained LoRA Weights
pipe.load_lora_weights("SaharAlhabsi/sdxl-base-1.1-archPlan")




# Check if CUDA (GPU) is available, else use CPU

# Load the base model (Stable Diffusion XL Base)
# base_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
# base_model = AutoModel.from_pretrained(base_model_name)
# base_model.to(device)

# Load the PEFT (LoRA) model configuration and weights
# lora_model_name = "SaharAlhabsi/sdxl-base-1.1-archPlan"

# peft_config = PeftConfig.from_pretrained(lora_model_name)
# lora_model = PeftModel(base_model, peft_config)
# lora_model.to(device)

# Alternatively, you can use StableDiffusionPipeline if that works better in your use case:
# pipe = StableDiffusionPipeline.from_pretrained(base_model_name)
# pipe.to(device)

# Enable PEFT (LoRA) weights
# pipe.load_lora_weights(lora_model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Extract user inputs from form
    building_area = request.form['building_area']
    room_count = int(request.form['room_count'])
    rooms = request.form.getlist('rooms[]')
    has_kitchen = 'kitchen' in request.form
    has_store = 'store' in request.form
    toilet_count = request.form['toilet_count']
    
    # Build the prompt for Stable Diffusion
    prompt = f"Home floor plan with {building_area} sqft, {room_count} rooms, including {', '.join(rooms)}."
    if has_kitchen:
        prompt += " Include a kitchen."
    if has_store:
        prompt += " Include a store room."
    prompt += f" {toilet_count} toilets."

    # Generate the image using the model (with LoRA weights)
    image = pipe(prompt).images[0]
    
    # Save the generated image to a file
    image_path = "static/floor_plan.png"
    image.save(image_path)

    # Send the image file as a response to the user
    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
