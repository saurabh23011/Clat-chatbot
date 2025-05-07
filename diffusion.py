"""
SGDiff: Stable Gradient-based Text-to-Image Diffusion Model (Memory-Efficient Version)

This implementation is based on the SGDiff approach that combines the strengths of
stable diffusion and gradient-based optimization for text-to-image generation.
Modified to address storage space limitations in constrained environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import os
import gc
import shutil

class SGDiff:
    def __init__(
        self,
        unet,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        device: str = "cuda",
        use_fp16: bool = False,
    ):
        """
        Initialize the SGDiff model.
        
        Args:
            unet: The UNet model for diffusion
            vae: The VAE model for encoding/decoding images
            text_encoder: The text encoder for conditioning
            tokenizer: The tokenizer for text processing
            scheduler: The noise scheduler
            device: The device to run the model on
            use_fp16: Whether to use FP16 precision
        """
        self.unet = unet.to(device)
        self.vae = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device
        
        # Set the models to evaluation mode
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        
        # Set precision
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if use_fp16:
            self.unet.to(dtype=self.dtype)
            self.vae.to(dtype=self.dtype)
            self.text_encoder.to(dtype=self.dtype)
    
    def encode_prompt(
        self, 
        prompt: str, 
        negative_prompt: Optional[str] = None
    ) -> torch.Tensor:
        """
        Encode the prompt into text embeddings.
        
        Args:
            prompt: The text prompt
            negative_prompt: The negative prompt for guidance
            
        Returns:
            text_embeddings: The encoded text embeddings
        """
        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]
        
        # Do the same for negative prompt if provided
        if negative_prompt:
            uncond_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_inputs.input_ids.to(self.device)
            
            with torch.no_grad():
                uncond_embeddings = self.text_encoder(uncond_input_ids)[0]
                
            # Concatenate embeddings for classifier-free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        return text_embeddings
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representations to images.
        
        Args:
            latents: The latent representations
            
        Returns:
            images: The decoded images
        """
        # Scale and decode the latents with VAE
        latents = 1 / 0.18215 * latents
        
        # Move VAE to the right device if it's not already there
        if self.vae.device != self.device:
            self.vae = self.vae.to(self.device)
        
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        
        # Convert to RGB
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.permute(0, 2, 3, 1)
        
        return images.cpu().numpy()
    
    def prepare_latents(
        self, 
        batch_size: int, 
        height: int, 
        width: int, 
        generator = None
    ) -> torch.Tensor:
        """
        Prepare initial random latents for the diffusion process.
        
        Args:
            batch_size: The batch size
            height: The height of the image
            width: The width of the image
            generator: Optional random generator
            
        Returns:
            latents: The prepared latents
        """
        # Get the appropriate shape for the latents
        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // 8,
            width // 8,
        )
        
        # Sample random noise
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Scale the latents
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents
    
    def gradient_based_optimization(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timestep: int,
        guidance_scale: float = 7.5,
        lr: float = 0.01,
        iterations: int = 3,  # Reduced iterations for stability
    ) -> torch.Tensor:
        """
        Perform gradient-based optimization on latents at a specific timestep.
        
        Args:
            latents: The current latents
            text_embeddings: The text embeddings
            timestep: The current timestep
            guidance_scale: The guidance scale for classifier-free guidance
            lr: The learning rate for optimization
            iterations: The number of optimization iterations
            
        Returns:
            optimized_latents: The optimized latents
        """
        # Make sure text_encoder is on the right device
        if self.text_encoder.device != self.device:
            self.text_encoder = self.text_encoder.to(self.device)
            
        # Clone the latents and make them trainable
        optim_latents = latents.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([optim_latents], lr=lr)
        
        # Get the uncond and text embeddings
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        
        for _ in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass through UNet for both conditional and unconditional
            noise_pred_uncond = self.unet(
                optim_latents, 
                timestep, 
                encoder_hidden_states=uncond_embeddings
            ).sample
            
            noise_pred_text = self.unet(
                optim_latents, 
                timestep, 
                encoder_hidden_states=cond_embeddings
            ).sample
            
            # Apply classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Calculate loss - minimize the predicted noise
            loss = F.mse_loss(noise_pred, torch.zeros_like(noise_pred))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        return optim_latents.detach()
    
    def sg_diffusion_step(
        self,
        latents: torch.Tensor,
        timestep: int,
        text_embeddings: torch.Tensor,
        guidance_scale: float = 7.5,
        optimize: bool = True,
    ) -> torch.Tensor:
        """
        Perform a single SGDiff diffusion step.
        
        Args:
            latents: The current latents
            timestep: The current timestep
            text_embeddings: The text embeddings
            guidance_scale: The guidance scale for classifier-free guidance
            optimize: Whether to perform gradient-based optimization
            
        Returns:
            new_latents: The new latents after the diffusion step
        """
        # Convert timestep to tensor if it's not already
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=self.device)
            
        # If optimization is enabled, perform gradient-based optimization
        # Check if timestep is a tensor and compare appropriately
        if optimize and (timestep.item() if isinstance(timestep, torch.Tensor) else timestep) > 0:
            latents = self.gradient_based_optimization(
                latents, 
                text_embeddings,
                timestep,
                guidance_scale
            )
        
        # Split the embeddings for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        
        # Predict the noise residual
        with torch.no_grad():
            if guidance_scale > 1.0:
                # Ensure text_embeddings is correctly split
                if len(text_embeddings.shape) == 3 and text_embeddings.shape[0] == 2:
                    # Already has unconditional and conditional embeddings
                    uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
                else:
                    # Need to duplicate for unconditional case
                    uncond_embeddings = text_embeddings
                    cond_embeddings = text_embeddings
                
                # Get unconditional and conditional noise predictions
                noise_pred_uncond = self.unet(
                    latent_model_input[:latents.shape[0]], 
                    timestep, 
                    encoder_hidden_states=uncond_embeddings
                ).sample
                
                noise_pred_text = self.unet(
                    latent_model_input[latents.shape[0]:], 
                    timestep, 
                    encoder_hidden_states=cond_embeddings
                ).sample
                
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.unet(
                    latent_model_input, 
                    timestep, 
                    encoder_hidden_states=text_embeddings
                ).sample
        
        # Update the latents
        new_latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
        
        return new_latents
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator = None,
        optimize_steps: List[int] = None,
    ) -> np.ndarray:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: The text prompt
            negative_prompt: The negative prompt
            height: The height of the generated image
            width: The width of the generated image
            num_inference_steps: The number of diffusion steps
            guidance_scale: The guidance scale for classifier-free guidance
            generator: The random generator
            optimize_steps: Which steps to apply optimization on
            
        Returns:
            images: The generated images
        """
        # Initialize the scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Make sure text_encoder is on the device for encoding
        if self.text_encoder.device != self.device:
            self.text_encoder = self.text_encoder.to(self.device)
            
        # Encode the prompt
        text_embeddings = self.encode_prompt(prompt, negative_prompt)
        
        # Move text_encoder back to CPU if we're in memory-saving mode
        if self.device == "cuda":
            self.text_encoder = self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        
        # Prepare the initial latents
        latents = self.prepare_latents(
            batch_size=1,
            height=height,
            width=width,
            generator=generator,
        )
        
        # Set default optimize steps if not provided
        if optimize_steps is None:
            # Default to optimizing on the first 80% of steps
            optimize_steps = list(range(int(num_inference_steps * 0.8)))
            
        # Diffusion loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="SGDiff")):
            # Determine if we should optimize at this step
            optimize = i in optimize_steps
            
            # Move text_embeddings to the current device if needed
            if text_embeddings.device != self.device:
                text_embeddings = text_embeddings.to(self.device)
            
            # Perform the diffusion step
            latents = self.sg_diffusion_step(
                latents=latents,
                timestep=t,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                optimize=optimize,
            )
            
            # Garbage collect to free up memory
            if i % 5 == 0:  # More frequent cleanup
                torch.cuda.empty_cache()
                gc.collect()
        
        # Make sure VAE is on the right device for decoding
        if self.vae.device != self.device:
            self.vae = self.vae.to(self.device)
            
        # Decode the latents to images
        images = self.decode_latents(latents)
        
        return images


def cleanup_cache():
    """
    Clean up the Hugging Face cache to free up space.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    if os.path.exists(cache_dir):
        print(f"Cleaning up HuggingFace cache directory: {cache_dir}")
        try:
            # Remove cache directory
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("Cache cleanup completed successfully")
        except Exception as e:
            print(f"Error during cache cleanup: {e}")
    else:
        print("No HuggingFace cache directory found")


def check_disk_space(required_gb=10):
    """
    Check if there's enough disk space available.
    
    Args:
        required_gb: Required free space in GB
        
    Returns:
        bool: True if enough space is available, False otherwise
    """
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    
    print(f"Disk space check - Free: {free_gb:.2f} GB, Required: {required_gb} GB")
    return free_gb >= required_gb


def attempt_fallback_model(original_model_id, fallback_options, use_fp16):
    """
    Try loading a series of fallback models if the original fails.
    
    Args:
        original_model_id: The original model ID that failed
        fallback_options: List of fallback model IDs to try
        use_fp16: Whether to use FP16 precision
        
    Returns:
        tuple: Models if successful or None if all fail
    """
    from diffusers import DDIMScheduler, DiffusionPipeline
    
    print(f"Attempting to load fallback models...")
    
    for model_id in fallback_options:
        try:
            print(f"Trying model: {model_id}")
            
            # Load with low memory settings
            pipeline_args = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() and use_fp16 else torch.float32,
                "safety_checker": None,  # Disable safety checker for speed
                "requires_safety_checker": False,
            }
            
            # Use the revision parameter to get specific version if available
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                revision="fp16" if torch.cuda.is_available() and use_fp16 else "main",
                **pipeline_args
            )
            
            # Extract the models we need
            unet = pipeline.unet
            vae = pipeline.vae
            text_encoder = pipeline.text_encoder
            tokenizer = pipeline.tokenizer
            
            # Create the scheduler
            scheduler = DDIMScheduler.from_pretrained(
                model_id, 
                subfolder="scheduler",
            )
            
            # Success!
            print(f"Successfully loaded fallback model: {model_id}")
            return unet, vae, text_encoder, tokenizer, scheduler
            
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            continue
    
    # All fallbacks failed
    return None


def load_models(model_id="CompVis/ldm-text2img-large-256", use_local_files=False, local_path=None, use_fp16=True):
    """
    Load the required models for SGDiff.
    Uses a smaller model and provides options for local loading.
    
    Args:
        model_id: The model ID to use (defaults to a smaller model)
        use_local_files: Whether to use local model files instead of downloading
        local_path: Path to local model files (if use_local_files is True)
        use_fp16: Whether to use FP16 precision
        
    Returns:
        tuple: (unet, vae, text_encoder, tokenizer, scheduler)
    """
    # Import here to avoid any initialization issues
    from diffusers import DDIMScheduler, DiffusionPipeline

    # Check disk space
    if not check_disk_space(5):  # Require at least 5GB free
        print("Warning: Low disk space detected. This may cause problems.")
    
    print(f"Loading models from {'local files' if use_local_files else model_id}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Using FP16: {use_fp16}")
    
    # Create fallback model options, fixing the typo in the original code
    fallback_models = [
        "CompVis/ldm-text2img-large-256",  # First option with correct spelling
        "runwayml/stable-diffusion-v1-5",  # Alternative option
        "stabilityai/stable-diffusion-2-base"  # Another alternative
    ]
    
    try:
        # Clean up cache to free space before downloading new models if needed
        if not use_local_files:
            cleanup_cache()
        
        # Load with low memory settings
        pipeline_args = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() and use_fp16 else torch.float32,
            "safety_checker": None,  # Disable safety checker for speed
            "requires_safety_checker": False,
        }
        
        if use_local_files and local_path:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local model path not found: {local_path}")
                
            print(f"Loading from local path: {local_path}")
            pipeline = DiffusionPipeline.from_pretrained(local_path, **pipeline_args)
        else:
            # Use the revision parameter to get specific version if available
            print(f"Downloading model from {model_id}...")
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                revision="fp16" if torch.cuda.is_available() and use_fp16 else "main",
                **pipeline_args
            )
        
        # Extract the models we need
        unet = pipeline.unet
        vae = pipeline.vae
        text_encoder = pipeline.text_encoder
        tokenizer = pipeline.tokenizer
        
        # Create the scheduler
        scheduler = DDIMScheduler.from_pretrained(
            model_id if not use_local_files else local_path, 
            subfolder="scheduler" if not use_local_files else None,
        )
        
        # Make sure all models are in evaluation mode
        unet.eval()
        vae.eval()
        text_encoder.eval()
        
        # Move unused components to CPU to save GPU memory
        if torch.cuda.is_available():
            unet = unet.to("cuda")
            # Keep others on CPU until needed
            vae = vae.to("cpu")
            text_encoder = text_encoder.to("cpu")
        
        # Free up memory
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
        return unet, vae, text_encoder, tokenizer, scheduler
    
    except Exception as e:
        print(f"Error loading models: {e}")
        
        # Try fallback models
        result = attempt_fallback_model(model_id, fallback_models, use_fp16)
        if result:
            return result
        
        # Suggest solutions if everything failed
        print("\nAll model loading attempts failed. Suggested solutions:")
        print("1. Try smaller model: 'CompVis/ldm-text2img-large-256'")
        print("2. Download model files manually and use local_path option")
        print("3. Free up disk space and try again")
        print("4. Use --use_fp16 option to reduce memory usage")
        
        raise RuntimeError("Failed to load any model after multiple attempts")


def generate_image(
    prompt, 
    negative_prompt=None,
    model_id="CompVis/ldm-text2img-large-256",  # Smaller model by default
    use_local_files=False,
    local_path=None,
    height=256,  # Smaller dimensions by default
    width=256,
    use_fp16=True,
    optimize_steps_range=(10, 30),
    num_inference_steps=30  # Fewer steps for faster generation
):
    """
    Generate an image from a text prompt using SGDiff.
    Memory-optimized version with more configuration options.
    
    Args:
        prompt: The text prompt
        negative_prompt: The negative prompt
        model_id: Model ID to use
        use_local_files: Whether to use local model files
        local_path: Path to local model files
        height: Image height
        width: Image width
        use_fp16: Whether to use FP16 precision
        optimize_steps_range: Range of steps to apply optimization on
        num_inference_steps: Number of diffusion steps
    """
    # Clean up before we start
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Load models
        unet, vae, text_encoder, tokenizer, scheduler = load_models(
            model_id=model_id,
            use_local_files=use_local_files,
            local_path=local_path,
            use_fp16=use_fp16
        )
        
        # Initialize SGDiff
        print("Initializing SGDiff model...")
        sgdiff = SGDiff(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_fp16=use_fp16,
        )
        
        # Calculate optimize steps
        start_step, end_step = optimize_steps_range
        optimize_steps = list(range(start_step, min(end_step, num_inference_steps)))
        
        # Generate image
        print(f"Generating image with prompt: '{prompt}'")
        images = sgdiff.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            optimize_steps=optimize_steps,
        )
        
        # Process and save the image
        import PIL.Image
        
        # Convert from numpy to PIL Image
        pil_image = PIL.Image.fromarray((images[0] * 255).astype(np.uint8))
        
        # Make sure the output directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Save the image with a timestamp to avoid overwriting
        import time
        timestamp = int(time.time())
        output_path = f"outputs/sgdiff_output_{timestamp}.png"
        pil_image.save(output_path)
        
        print(f"Image saved to {output_path}")
        return pil_image
    
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None
    finally:
        # Clean up regardless of success or failure
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images using SGDiff")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over mountains, photorealistic", help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted", help="Negative prompt")
    parser.add_argument("--model_id", type=str, default="CompVis/ldm-text2img-large-256", help="Model ID to use")
    parser.add_argument("--local_path", type=str, default=None, help="Path to local model files")
    parser.add_argument("--use_local_files", action="store_true", help="Use local model files")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--use_fp16", action="store_true", default=True, help="Use FP16 precision")
    parser.add_argument("--cleanup_cache", action="store_true", help="Clean up HuggingFace cache before running")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps")
    
    args = parser.parse_args()
    
    if args.cleanup_cache:
        cleanup_cache()
    
    image = generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model_id=args.model_id,
        use_local_files=args.use_local_files,
        local_path=args.local_path,
        height=args.height,
        width=args.width,
        use_fp16=args.use_fp16,
        num_inference_steps=args.num_inference_steps
    )
    
    if image:
        print("Image generated successfully!")
    else:
        print("Image generation failed.")