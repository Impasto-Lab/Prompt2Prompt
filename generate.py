"""Prompt-to-Prompt word swap (Hertz et al., 2023) on the Latent Diffusion Model (Rombach et al., 2022).

The first prompt is the source. In the first --inject-steps steps, every other
prompt uses the source's cross-attention maps, aligned word by word.

    prompt ──tokenizer──▶ token ids ──LDMBert──▶ text features ───────────────┐
                                                                               ▼
    random latent ──▶ [ U-Net predicts noise ─▶ guidance ─▶ DDIM update ] × steps
                                                                               │
    PNG ◀── pixels ◀── autoencoder.decode ◀── clean latent ◀───────────────────┘
"""
import argparse
import os
from pathlib import Path

# Let PyTorch run ops that Apple's MPS backend lacks on the CPU instead of failing.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from utils.align import token_mapper  # noqa: E402
from utils.config import LATENT_SCALING_FACTOR, ROOT  # noqa: E402
from utils.device import pick_device, pick_dtype  # noqa: E402
from utils.image import make_grid, to_pil  # noqa: E402
from utils.loader import load_models  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps, 1-999")
    parser.add_argument("--guidance", type=float, default=5.0, help="classifier-free guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM noise: 0 = deterministic, 1 = DDPM-like")
    parser.add_argument("--seed", type=int, default=42, help="one seed shared by all prompts")
    parser.add_argument("--batch-size", type=int, default=8, help="prompts sampled together")
    parser.add_argument(
        "--inject-steps", type=int, default=20,
        help="first n steps (0 to steps) in which every prompt uses the first prompt's cross-attention maps",
    )
    parser.add_argument("--width", type=int, default=256, help="multiple of 64")
    parser.add_argument("--height", type=int, default=256, help="multiple of 64")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, cuda:1, ...")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--output", type=Path, help="grid PNG path (default: outputs/cakes-seed<seed>-inject<n>.png)",
    )
    args = parser.parse_args()
    # Check the values here, before the 6 GB checkpoint loads.
    if not 1 <= args.steps <= 999:
        parser.error("--steps must be in [1, 999].")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.eta < 0:
        parser.error("--eta must not be negative.")
    if not 0 <= args.inject_steps <= args.steps:
        parser.error("--inject-steps must be in [0, steps].")
    # The autoencoder shrinks images 8x, then the U-Net halves the latent 3 more
    # times. Its skip connections only line up if every halving is exact: 8 * 2**3 = 64.
    if args.width % 64 or args.height % 64:
        parser.error("--width and --height must be multiples of 64.")
    return args


@torch.no_grad()
def encode_text(prompts, tokenizer, text_encoder, device):
    ids = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    return text_encoder(ids.to(device))


@torch.no_grad()
def generate(prompts, models, args, device, dtype, record=False, source=None):
    """Sample one image per prompt. Every prompt starts from the same seeded noise.

    Cross-attention maps are lists indexed ``[step][layer]``, each of the text
    half only, [batch, heads, h*w, 77]. With ``record``, return this run's maps.
    With ``source = (source_prompt, source_maps)`` (maps of batch 1), the first
    ``args.inject_steps`` steps use the source maps, aligned word by word to
    each prompt's tokens, in place of this run's own maps at the same step and layer.

    Returns ``(images, maps)``; ``maps`` is empty without ``record``.
    """
    tokenizer, text_encoder, unet, autoencoder, scheduler = models
    batch = len(prompts)

    # Classifier-free guidance: run the U-Net on "" and on the prompt in one batch.
    context = encode_text([""] * batch + prompts, tokenizer, text_encoder, device)

    generator = torch.Generator("cpu").manual_seed(args.seed)
    shape = (1, unet.config.in_channels, args.height // 8, args.width // 8)
    latents = torch.randn(shape, generator=generator).expand(batch, -1, -1, -1)
    latents = latents.to(device=device, dtype=dtype)

    if source is not None:
        source_prompt, source_maps = source
        # [batch, 1, 77, 77]: moves source token columns to each prompt's token positions.
        mappers = torch.stack([token_mapper(source_prompt, p, tokenizer) for p in prompts])
        mappers = mappers[:, None].to(device=device, dtype=dtype)

    scheduler.set_timesteps(args.steps)
    attention_maps = []
    for step, t in enumerate(tqdm(scheduler.timesteps, desc="DDIM", leave=False)):
        edit = None
        if source is not None and step < args.inject_steps:
            # Keep the empty-prompt half; give the text half the source's aligned map.
            def edit(layer, probs):
                return torch.cat([probs[:batch], source_maps[step][layer] @ mappers])

        # maps: list of the image-text softmax(Q K^T) maps in the U-Net at step t.
        noise, maps = unet.forward_with_cross_attention(torch.cat([latents] * 2), t, context, edit)
        noise_uncond, noise_text = noise.chunk(2)
        noise_pred = noise_uncond + args.guidance * (noise_text - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents, eta=args.eta, generator=generator)
        if record:
            # clone(): a slice would keep the empty-prompt half in memory too.
            attention_maps.append([m[batch:].clone() for m in maps])

    return to_pil(autoencoder.decode(latents / LATENT_SCALING_FACTOR)), attention_maps


def main():
    # Same order as the example figure: two rows of eight.
    prompts = [
        "lemon cake.", "cheese cake.", "apple cake.", "pumpkin cake.",
        "chocolate cake.", "pistachio cake.", "jello cake.", "matcha cake.",
        "monster cake.", "lego cake.", "beet cake.", "pepperoni cake.",
        "fish cake.", "rice cake.", "pasta cake.", "brick cake.",
    ]
    cols = 8

    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype)
    models = load_models(device, dtype)

    # The first prompt is the source: record its maps at every step (kept in memory only).
    images, source_maps = generate(prompts[:1], models, args, device, dtype, record=True)
    for start in tqdm(range(1, len(prompts), args.batch_size), desc="batches"):
        batch_prompts = prompts[start:start + args.batch_size]
        images += generate(batch_prompts, models, args, device, dtype, source=(prompts[0], source_maps))[0]

    output = args.output or ROOT / "outputs" / f"cakes-seed{args.seed}-inject{args.inject_steps}.png"
    single_dir = output.with_suffix("")
    single_dir.mkdir(parents=True, exist_ok=True)
    for i, (image, prompt) in enumerate(zip(images, prompts), 1):
        image.save(single_dir / f"{i:02d}-{prompt.rstrip('.').replace(' ', '-')}.png")

    make_grid(images, [f"“{p}”" for p in prompts], cols).save(output)
    print(f"Saved {output} and {len(images)} single images in {single_dir}")


if __name__ == "__main__":
    main()
