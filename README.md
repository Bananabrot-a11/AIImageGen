# AIImageGen
Image Generator using AI

---

## How To Run

1. pip install -r requirements.txt
2. python MultiImage.py (in directory folder)

- you need to add your Hugging Face token at  login("hf_...")  

---

## Specifications

the current model (dreamlike-art/dreamlike-photoreal-2.0) is landscape-optimized.
AI model can easily be changed by changing out the name in  {{model_id}} to a different model.
Default model is runwayml/stable-diffusion-v1-5, which is a general-purpose model (okay but not really great).

---

## Settings

Current settings

```python
width = 512
height = 512
num_inference_steps = 200
guidance_scale = 7.5
```
width and height must be divisible by 8

high num_inference_steps = good quality image but long runtime
low num_inference_steps = fast image generation but low quality

guidance_scale = how closely the prompt should be followed.

To change the amount of images generated, edit the  num_images= variable.

