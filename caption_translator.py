import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

BLIP_PATH = r"E:\Siva\blip_project\blip_final_model"
NLLB_PATH = r"E:\Siva\models\nllb_600"

_blip_processor = None
_blip_model = None
_nllb_tokenizer = None
_nllb_model = None


# --------------------------
# BLIP MODEL
# --------------------------
def get_blip_models():
    global _blip_processor, _blip_model

    if _blip_processor is None:
        print("Loading BLIP...")
        _blip_processor = BlipProcessor.from_pretrained(
            BLIP_PATH,
            local_files_only=True
        )

        _blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_PATH,
            local_files_only=True
        ).to(device)

        _blip_model.eval()

    return _blip_processor, _blip_model


def generate_caption(image_path):
    try:
        blip_processor, blip_model = get_blip_models()

        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt").to(device)

        with torch.inference_mode():
            output = blip_model.generate(
                **inputs,
                max_length=40,
                num_beams=3,
                do_sample=False
            )

        caption = blip_processor.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        return caption

    except Exception as e:
        print("Caption Error:", e)
        return None


# --------------------------
# NLLB TRANSLATION MODEL
# --------------------------

LANG_CODES = {
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ml": "mal_Mlym"
}


def get_nllb_models():
    global _nllb_tokenizer, _nllb_model

    if _nllb_tokenizer is None:
        print("Loading NLLB...")

        _nllb_tokenizer = AutoTokenizer.from_pretrained(
            NLLB_PATH,
            local_files_only=True
        )

        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_PATH,
            local_files_only=True
        ).to(device)

        _nllb_model.eval()

    return _nllb_tokenizer, _nllb_model


def translate_nllb(text, lang):
    try:
        if not text or not isinstance(text, str):
            return None

        nllb_tokenizer, nllb_model = get_nllb_models()

        tgt = LANG_CODES.get(lang)
        if not tgt:
            return None

        # Set source language
        nllb_tokenizer.src_lang = "eng_Latn"

        inputs = nllb_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        translated_tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(tgt),
            max_length=80
        )

        translated = nllb_tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]

        return translated

    except Exception as e:
        print("Translation Error:", e)
        return None


# --------------------------
# POLISHING
# --------------------------

def safe_replace(text, old):
    if not text:
        return text
    return text.replace(old, "").strip()


POLISH = {
    "ta": lambda t: safe_replace(t, "ஒரு "),
    "te": lambda t: safe_replace(t, "ఒక "),
    "hi": lambda t: safe_replace(t, "एक "),
    "kn": lambda t: safe_replace(t, "ಒಂದು "),
    "ml": lambda t: safe_replace(t, "ഒരു "),
}


def translate_single(text, lang):
    if lang not in LANG_CODES:
        return {lang: "Unsupported language"}

    raw = translate_nllb(text, lang)

    if raw is None:
        return {lang: "Translation failed"}

    clean = POLISH.get(lang, lambda x: x)(raw)

    return {lang: clean}
