import base64
import json
import shutil
import regex as re
import torch
import os
from datetime import datetime
from transformers import AutoTokenizer
from translation_engine.model.BIALinguaNet import BIALinguaNet


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BIALinguaNet(**checkpoint["init_args"])
    model.to(device)
    model.eval()

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as e:
        print(f"Failed to load all parameters: {e}")

    return model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(
        {"pad_token": "<PAD>", "bos_token": "<SOS>", "eos_token": "<EOS>"}
    )
    return tokenizer


def add_background_image(image_path, st):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state(st):
    if "device" not in st.session_state:
        st.session_state.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if "en_dr_model" not in st.session_state:
        en_dr_checkpoint_path = (
            "./translation_engine/src/checkpoints/en_dr_chpts.pth.tar"
        )
        st.session_state.en_dr_model = load_model(
            en_dr_checkpoint_path, st.session_state.device
        )

    if "dr_en_model" not in st.session_state:
        dr_en_checkpoint_path = (
            "./translation_engine/src/checkpoints/dr_en_chpts.pth.tar"
        )
        st.session_state.dr_en_model = load_model(
            dr_en_checkpoint_path, st.session_state.device
        )

    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = load_tokenizer()
