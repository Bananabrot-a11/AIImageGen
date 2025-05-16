# AIImageGen
Image Generator using AI

---

# How To Run

1. pip install -r requirements.txt
2. python MultiImage.py (in directory Folder)

:::info
    you need to add you'r Huggingface Token at  login("hf_...") 
:::

---

# Spesifications

the curent Model (dreamlike-art/dreamlike-photoreal-2.0) Is Landscape-optimized.
AI- Model can easely be changed by chnaging out the name in  {{model_id}} To a diferent Model.
Default Model is runwayml/stable-diffusion-v1-5 Wich is a general purpous Model (Ok but not realy great).

---

# Settings

Curent Settings 

```python
width=512
height=512
num_inference_steps=200
guidance_scale=7.5
```
width and height must be dividabel by 8


high num_inference_steps = Good Quality image but long runntime
low num_inference_steps = Fast Image generation but lov quality 

guidance_scale = How closly the Promt should be foloved.
