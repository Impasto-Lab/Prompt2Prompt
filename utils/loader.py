"""Read checkpoint configs and load pretrained weights into the local models."""
import json
from types import SimpleNamespace

import torch
from transformers import BertTokenizer

from models.autoencoder import AutoencoderKL
from models.bert import LDMBertModel
from models.scheduler import DDIMScheduler
from models.unet import ConditionalUNet
from utils.config import MODEL_DIR, MODEL_FILES


def check_model_files():
    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Model files missing. Run `python -m scripts.download_model` first.\n" + "\n".join(missing)
        )


def read_config(path):
    """Load a JSON config with attribute access: ``config.in_channels``."""
    return json.loads(path.read_text(), object_hook=lambda fields: SimpleNamespace(**fields))


def load_model(model_class, config, checkpoint, device, dtype):
    """Build ``model_class(config)`` and fill it with the checkpoint's weights.

    The model is created on the "meta" device (shapes only, no memory), then
    the checkpoint tensors are assigned directly. ``strict=True`` fails loudly
    if any checkpoint key and model attribute do not match one-to-one.
    """
    with torch.device("meta"):
        model = model_class(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=dtype)


def load_models(device, dtype):
    """Load every LDM component: tokenizer, text encoder, U-Net, autoencoder, scheduler."""
    check_model_files()
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR / "tokenizer", local_files_only=True)

    text_encoder = load_model(
        LDMBertModel, read_config(MODEL_DIR / "bert" / "config.json"),
        MODEL_DIR / "bert" / "pytorch_model.bin", device, dtype,
    )

    unet_config = read_config(MODEL_DIR / "unet" / "config.json")
    # Cross-attention reads the text features, so its input width must equal the
    # text encoder's output width (1280). The converted config leaves it out.
    unet_config.cross_attention_dim = text_encoder.config.d_model
    unet = load_model(
        ConditionalUNet, unet_config, MODEL_DIR / "unet" / "diffusion_pytorch_model.bin", device, dtype,
    )

    # The published folder is named "vqvae", but it holds a KL autoencoder.
    autoencoder = load_model(
        AutoencoderKL, read_config(MODEL_DIR / "vqvae" / "config.json"),
        MODEL_DIR / "vqvae" / "diffusion_pytorch_model.bin", device, dtype,
    )

    scheduler = DDIMScheduler(read_config(MODEL_DIR / "scheduler" / "scheduler_config.json"))
    return tokenizer, text_encoder, unet, autoencoder, scheduler
