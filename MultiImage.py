import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
import os
from colorama import Fore, Style, init
import time
import platform
import psutil
import traceback
from rich.console import Console

init(autoreset=True)
console = Console()
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def check_system():
    """Comprehensive hardware diagnostics"""
    console.rule("[bold yellow]⚙️ System Diagnostics")
    

    print(f"{Fore.CYAN}OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total/1024**3:.1f} GB")


    if torch.cuda.is_available():
        print(f"\n{Fore.GREEN}✓ GPU Detected")
        print(f"{Fore.BLUE}→ {torch.cuda.get_device_name(0)}")
        print(f"→ CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
        print(f"→ VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        print(f"\n{Fore.RED}✗ No GPU Acceleration Available")
        print(f"{Fore.YELLOW}Running on CPU (very slow)")

    console.rule()


def load_model(model_name="dreamlike-art/dreamlike-photoreal-2.0"):
    """Load the model with proper precision handling"""
    model_id = model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        console.print(f"[yellow]⌛ Loading model (device: {device}, dtype: {dtype}, model: {model_id})...")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype, 
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        if device == "cuda":
            pipe.enable_attention_slicing()
            try:
                pipe.enable_xformers_memory_efficient_attention()
                console.print("[green]✓ XFormers enabled")
            except:
                console.print("[yellow]⚠ XFormers not available")

        console.print(f"[green]✓ Model loaded successfully on {device.upper()}")
        return pipe

    except Exception as e:
        console.print(f"[red]✗ Model loading failed: {str(e)}")
        raise


def generate_images(prompt, num_images=1, model_name="dreamlike-art/dreamlike-photoreal-2.0"):
    try:
        torch.cuda.empty_cache()
        pipe = load_model(model_name)
        
        console.print(f"[magenta]🎨 Generating {num_images} images for: {prompt[:50]}...")
        
        os.makedirs("outputs", exist_ok=True)
        prompt_folder = f"outputs/{prompt[:30].replace(' ', '_')}_{int(time.time())}"
        os.makedirs(prompt_folder, exist_ok=True)
        
        for i in range(num_images):
            image = pipe(
                prompt=prompt,
                width=512,
                height=512,
                num_inference_steps=200,
                guidance_scale=7.5
            ).images[0]
            filename = f"{prompt_folder}/{i+1}.png"
            image.save(filename)
            console.print(f"[green]✓ Saved: {filename}")
            
            time.sleep(1)
            
    except Exception as e:
        console.print(f"[red]✗ Generation failed: {e}")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        check_system()
        token = input("Enter your Hugging Face token (starts with 'hf_'): ").strip()
        login(token)
        prompt = input("Enter your image prompt: ")
        width = int(input("Image width (divisible by 8, default 512): ") or 512)
        height = int(input("Image height (divisible by 8, default 512): ") or 512)
        num_inference_steps = int(input("Number of inference steps (default 200): ") or 200)
        guidance_scale = float(input("Guidance scale (default 7.5): ") or 7.5)
        num_images = int(input("How many images to generate? (default 1): ") or 1)
        available_models = [
            ("dreamlike-art/dreamlike-photoreal-2.0", "Photorealistic, landscape-optimized, high detail"),
            ("runwayml/stable-diffusion-v1-5", "General-purpose, versatile, good for most prompts"),
            ("stabilityai/stable-diffusion-2-1", "General-purpose, improved over v1-5, more modern"),
            ("prompthero/openjourney", "Artistic, anime, and illustration style images"),
            ("nitrosocke/redshift-diffusion", "3D render, cinematic, Redshift-style visuals")
        ]
        print("\nAvailable AI Models:")
        for idx, (model, desc) in enumerate(available_models, 1):
            print(f"  {idx}. {model}\n     → {desc}")
        while True:
            try:
                model_choice = int(input(f"Select model [1-{len(available_models)}] (default 1): ") or 1)
                if 1 <= model_choice <= len(available_models):
                    model_name = available_models[model_choice - 1][0]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_models)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        def generate_images_custom(prompt, num_images, model_name, width, height, num_inference_steps, guidance_scale):
            try:
                torch.cuda.empty_cache()
                pipe = load_model(model_name)
                console.print(f"[magenta]🎨 Generating {num_images} images for: {prompt[:50]}... (Model: {model_name})")
                os.makedirs("outputs", exist_ok=True)
                prompt_folder = f"outputs/{prompt[:30].replace(' ', '_')}_{int(time.time())}"
                os.makedirs(prompt_folder, exist_ok=True)
                for i in range(num_images):
                    image = pipe(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    ).images[0]
                    filename = f"{prompt_folder}/{i+1}.png"
                    image.save(filename)
                    console.print(f"[green]✓ Saved: {filename}")
                    time.sleep(1)
            except Exception as e:
                console.print(f"[red]✗ Generation failed: {e}")
            finally:
                torch.cuda.empty_cache()

        generate_images_custom(prompt, num_images, model_name, width, height, num_inference_steps, guidance_scale)

    except Exception as e:
        console.print(f"{Fore.RED}💀 Fatal error: {e}")
        console.print(traceback.format_exc())
    finally:
        console.rule("[bold green]COMPLETED")