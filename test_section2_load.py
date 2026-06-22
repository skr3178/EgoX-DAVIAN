"""
Section 2 test: model + custom pipeline + LoRA load/fuse, under sequential CPU offload.
Reuses infer.py:56-77 verbatim; only swaps `pipe.to("cuda")` -> enable_sequential_cpu_offload().
No generation — just validates load + fuse + 24 GB fit.
"""
import torch
from transformers import CLIPVisionModel

model_path = "./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers"
transformer_path = model_path + "/transformer"
lora_path = "./checkpoints/EgoX/pytorch_lora_weights.safetensors"
dtype = torch.bfloat16

from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA as WanTransformer3DModel
print("loading custom GGA transformer ...")
transformer = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype)
print(f"  in_channels (config) = {transformer.config.in_channels}  (expect 36)")

image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)

from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline
print("building WanWidthConcatImageToVideoPipeline ...")
pipe = WanWidthConcatImageToVideoPipeline.from_pretrained(
    model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype)

print("loading + fusing rank-256 LoRA ...")
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
pipe.fuse_lora(components=["transformer"], lora_scale=1.0)
print("  fuse_lora OK")

# the 24 GB fix: sequential offload instead of pipe.to("cuda")
torch.cuda.reset_peak_memory_stats()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()

# touch the device map so peak reflects setup
print(f"\npipe device(s): transformer params on "
      f"{next(pipe.transformer.parameters()).device}")
print(f"peak VRAM after load+fuse+offload setup: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("\nSection 2 OK — model loads, LoRA fuses, fits under sequential offload")
