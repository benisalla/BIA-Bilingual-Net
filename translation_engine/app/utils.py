import base64
import os

import torch

from translation_engine.data.dataloader import build_tokenizer
from translation_engine.model.BIALinguaNet import BIALinguaNet

# paths relative to the package, so the app works from any working directory
ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(ENGINE_DIR, "src", "checkpoints")
STYLE_PATH = os.path.join(ENGINE_DIR, "app", "style", "style.css")
IMG_DIR = os.path.join(ENGINE_DIR, "src", "img")

MODEL_FILES = {
    "en_dr_model": "en_dr_chpts.pth.tar",  # English -> Darija
    "dr_en_model": "dr_en_chpts.pth.tar",  # Darija -> English
}


def load_model(checkpoint_path, device="cpu"):
    """Rebuild a BIALinguaNet from a checkpoint saved by training (weights + init args)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BIALinguaNet(**checkpoint["init_args"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_tokenizer():
    return build_tokenizer()


def add_background_image(image_path, st):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    with open(STYLE_PATH, "r") as css_file:
        css_content = css_file.read()
        css_content = css_content.replace("{encoded_image}", encoded_image)
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def init_session_state(st):
    """Load the tokenizer and both models once per session. A missing checkpoint is stored as None."""
    if "device" not in st.session_state:
        st.session_state.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    for key, file_name in MODEL_FILES.items():
        if key not in st.session_state:
            path = os.path.join(CHECKPOINTS_DIR, file_name)
            st.session_state[key] = (
                load_model(path, st.session_state.device) if os.path.exists(path) else None
            )

    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = load_tokenizer()
