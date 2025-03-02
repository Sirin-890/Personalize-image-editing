import os
import math
import itertools
import logging
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

from .dataset import TextualInversionDataset, create_dataloader
from .utils import freeze_params, save_progress

logger = logging.getLogger(__name__)

def train_textual_inversion(
    pretrained_model_name_or_path,
    train_data_dir,
    output_dir,
    placeholder_token,
    initializer_token,
    learnable_property="object",
    seed=42,
    resolution=512,
    train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=5e-04,
    scale_lr=True,
    max_train_steps=2000,
    save_steps=250,
    mixed_precision="fp16",
    gradient_checkpointing=True,
    center_crop=False,
    repeats=100,
    device="cuda"
):
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        
    # First, create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tokenizer and add placeholder token
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    
    # Add the placeholder token as a special token
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    
    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    
    # Load models
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    
    # Resize token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Initialize placeholder token embedding with initializer token embedding
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    
    # Freeze models, we only train the token embeddings
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)
    
    # Create dataset and dataloader
    train_dataset = TextualInversionDataset(
        data_root=train_data_dir,
        tokenizer=tokenizer,
        size=resolution,
        placeholder_token=placeholder_token,
        repeats=repeats,
        learnable_property=learnable_property,
        center_crop=center_crop,
        set="train",
    )
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision
    )
    
    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()
    
    # Create dataloader
    train_dataloader = create_dataloader(train_dataset, batch_size=train_batch_size)
    
    # Scale learning rate if needed
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )
    
    # Prepare for accelerator
    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )
    
    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move models to device and set proper mode
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    # Keep vae in eval mode and unet in train mode (for gradient checkpointing)
    vae.eval()
    unet.train()
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    # Log training info
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Start training
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    
    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)
                
                # Zero out gradients for tokens that aren't the placeholder token
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Check if we should update the progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % save_steps == 0:
                    save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, save_path)
            
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
        
        accelerator.wait_for_everyone()
    
    # Save the final model and embeddings
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)
        
        # Save the final embeddings
        save_path = os.path.join(output_dir, f"learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_id, accelerator, save_path)
        
        # Save token information
        with open(os.path.join(output_dir, "token_identifier.txt"), "w") as f:
            f.write(placeholder_token)
        with open(os.path.join(output_dir, "type_of_concept.txt"), "w") as f:
            f.write(learnable_property)
            
    return pipeline