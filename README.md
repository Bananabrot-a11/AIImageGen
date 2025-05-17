# AIImageGen
Image Generator using AI
---

## How To Run

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

2. Start the program:
   ```powershell
   python MultiImage.py
   ```

- You will be prompted in the terminal for your Hugging Face token, image prompt, image size, number of inference steps, guidance scale, number of images, image base name, output folder name, and to select an AI model from a list.
- All generated images will be saved in the `outputs/` folder, organized by your chosen folder name or prompt.

---

## Specifications

You can select from several AI models in the terminal interface. Each model is described in the selection menu. The default model is `runwayml/stable-diffusion-v1-5` (general-purpose). For photorealistic landscapes, use `dreamlike-art/dreamlike-photoreal-2.0`. For portraits, try `SG161222/Realistic_Vision_V4.0`. More models are available in the menu.

---

## Settings

You will be prompted for these settings in the terminal:

- `width` and `height` (must be divisible by 8)
- `num_inference_steps` (higher = better quality, slower)
- `guidance_scale` (how closely the prompt is followed)
- `num_images` (number of images to generate)
- `image_name_base` (base name for image files)
- `folder_name_base` (base name for output folder)
- Model selection (choose from a list)

Example default values:
```python
width = 512
height = 512
num_inference_steps = 200
guidance_scale = 7.5
```

---

## Output

- Images are saved in the `outputs/` folder, in a subfolder named after your chosen folder name or prompt.
- Each image is named as `<image_name_base>_1.png`, `<image_name_base>_2.png`, etc.

---

## Notes


---
