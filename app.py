#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese NER demo (Flask) with model switch:
- CRF (joblib): models/CRF_best.joblib
- BiLSTM-CRF (PyTorch): models/BiLSTM_CRF_best.pt

This file is designed to be easy to adapt to YOUR saved formats.
If your saved artifacts differ (common!), search for "TODO" below.

Run:
  pip install -r requirements.txt
  python app.py
Then open: http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from flask import Flask, render_template, request, jsonify

import joblib

# PyTorch is only needed if you use BiLSTM-CRF
import torch
import torch.nn as nn

# =========================
# 1) Utilities
# =========================

def simple_vi_tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for demo:
    - splits on whitespace
    - keeps punctuation as separate tokens (basic)
    If your pipeline uses word segmentation (VD: VnCoreNLP / underthesea),
    replace this with your exact preprocessing to match training.
    """
    text = text.strip()
    if not text:
        return []
    # Separate punctuation
    text = re.sub(r"([.,!?;:()\"“”'’\[\]{}])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ")

def group_spans(tokens: List[str], tags: List[str]) -> List[Dict[str, str]]:
    """
    Groups adjacent tokens with same entity label (BIO supported).
    Returns list of {"text": ..., "label": ...}
    """
    spans = []
    cur_words = []
    cur_label = "O"

    def flush():
        nonlocal cur_words, cur_label
        if not cur_words:
            return
        spans.append({"text": " ".join(cur_words), "label": cur_label})
        cur_words = []

    for tok, tag in zip(tokens, tags):
        if tag == "O":
            flush()
            spans.append({"text": tok, "label": "O"})
            cur_label = "O"
            cur_words = []
            continue

        # BIO
        if tag.startswith("B-"):
            flush()
            cur_label = tag[2:]
            cur_words = [tok]
        elif tag.startswith("I-"):
            lab = tag[2:]
            if cur_label == lab and cur_words:
                cur_words.append(tok)
            else:
                # broken I- => start new span
                flush()
                cur_label = lab
                cur_words = [tok]
        else:
            # non-BIO tagset
            if cur_label == tag and cur_words:
                cur_words.append(tok)
            else:
                flush()
                cur_label = tag
                cur_words = [tok]

    flush()
    return spans

# =========================
# 2) Model wrappers
# =========================

class PredictorBase:
    name: str

    def predict(self, tokens: List[str]) -> List[str]:
        raise NotImplementedError

@dataclass
class CRFPredictor(PredictorBase):
    """
    Expected joblib content (recommended):
      {
        "model": <crf model>,
        "feature_extractor": callable or object,
        "tag2id": {...} (optional),
        "id2tag": {...} (optional),
        "config": {...} (optional)
      }

    TODO: Adapt `featurize()` to match YOUR CRF features.
    """
    name: str
    bundle: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "CRFPredictor":
        bundle = joblib.load(path)
        return CRFPredictor(name="CRF", bundle=bundle)

    def featurize(self, tokens: List[str]) -> Any:
        # --- Option A: you saved a feature_extractor in the bundle ---
        fx = None
        if isinstance(self.bundle, dict):
            fx = self.bundle.get("feature_extractor", None)
        if fx is not None:
            # Many CRF libs expect list of dict features per token
            return fx(tokens)

        # --- Option B (fallback): use the same features as training ---
        # Based on sent2features from train_CRF.ipynb
        feats = []
        for i in range(len(tokens)):
            w = tokens[i]
            f = {
                "bias": 1.0,
                "w.lower": w.lower(),
                "w.isupper": w.isupper(),
                "w.istitle": w.istitle(),
                "w.isdigit": w.isdigit(),
                "w.len": len(w),
                "pref2": w[:2].lower(),
                "pref3": w[:3].lower(),
                "suf2": w[-2:].lower(),
                "suf3": w[-3:].lower(),
                "has_hyphen": "-" in w
            }
            # Previous word features
            if i > 0:
                wp = tokens[i-1]
                f["-1:w.lower"] = wp.lower()
                f["-1:w.istitle"] = wp.istitle()
            else:
                f["BOS"] = True

            # Next word features
            if i < len(tokens) - 1:
                wn = tokens[i+1]
                f["+1:w.lower"] = wn.lower()
                f["+1:w.istitle"] = wn.istitle()
            else:
                f["EOS"] = True
            
            feats.append(f)
        return feats

    def predict(self, tokens: List[str]) -> List[str]:
        # Check if bundle is a dict or the model itself
        if isinstance(self.bundle, dict):
            model = self.bundle.get("model", None)
            if model is None:
                model = self.bundle
        else:
            # bundle is the model itself
            model = self.bundle

        X = self.featurize(tokens)

        # sklearn-crfsuite style: model.predict([X])[0]
        if hasattr(model, "predict"):
            y = model.predict([X])[0]
            return list(map(str, y))

        raise RuntimeError("CRF bundle/model doesn't expose a .predict API. Adapt CRFPredictor.predict().")


class BiLSTMCRF(nn.Module):
    """
    Minimal BiLSTM-CRF skeleton.
    You MUST align this to your training architecture, embeddings, hidden sizes, etc.

    Recommended checkpoint format (dict):
      {
        "model_state": state_dict,
        "word2id": {...},
        "id2tag": {...} or "tag2id": {...},
        "config": {"emb_dim":..., "hidden_dim":..., ...}
      }
    If your .pt is only state_dict, you must provide vocab + tag maps elsewhere.
    """
    def __init__(self, vocab_size: int, tagset_size: int, emb_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

        # Optional: torchcrf
        try:
            from torchcrf import CRF  # type: ignore
            self.crf = CRF(tagset_size, batch_first=True)
        except Exception as e:
            raise RuntimeError(
                "torchcrf is required for BiLSTM-CRF. Install: pip install pytorch-crf"
            ) from e

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h, _ = self.lstm(h)
        emissions = self.fc(h)
        # decode expects mask as byte/bool
        return emissions

    def decode(self, x: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask.bool())

@dataclass
class BiLSTMCRFPredictor(PredictorBase):
    name: str
    model: BiLSTMCRF
    word2id: Dict[str, int]
    id2tag: Dict[int, str]
    device: str = "cpu"

    @staticmethod
    def load(path: str, device: Optional[str] = None) -> "BiLSTMCRFPredictor":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(path, map_location="cpu")

        # ------ Parse checkpoint ------
        if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt):
            state = ckpt.get("model_state", ckpt.get("state_dict"))
            word2id = ckpt.get("word2id") or ckpt.get("vocab") or ckpt.get("word_to_idx")
            tag2id = ckpt.get("tag2id") or ckpt.get("tag_to_idx")
            id2tag = ckpt.get("id2tag")
            config = ckpt.get("config", {})

            if word2id is None or (id2tag is None and tag2id is None):
                raise RuntimeError(
                    "BiLSTM-CRF checkpoint is missing word2id and tag mapping.\n"
                    "✅ Fix: save a checkpoint dict containing word2id + id2tag (or tag2id).\n"
                    "See README.md in this folder."
                )

            if id2tag is None:
                id2tag = {int(v): k for k, v in tag2id.items()}  # type: ignore

            emb_dim = int(config.get("emb_dim", 128))
            hidden_dim = int(config.get("hidden_dim", 256))

            model = BiLSTMCRF(
                vocab_size=len(word2id),
                tagset_size=len(id2tag),
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
            )
            model.load_state_dict(state)
        else:
            # If user saved only the full model object
            if hasattr(ckpt, "decode"):
                raise RuntimeError(
                    "It looks like you saved a full model object. Great — but this demo expects a dict checkpoint.\n"
                    "✅ Quick fix: save a dict with model_state + word2id + id2tag.\n"
                    "Or adapt BiLSTMCRFPredictor.load() to your format."
                )
            raise RuntimeError("Unsupported .pt format for BiLSTM-CRF checkpoint.")

        model.to(device)
        model.eval()

        return BiLSTMCRFPredictor(name="BiLSTM-CRF", model=model, word2id=word2id, id2tag=id2tag, device=device)

    def encode(self, tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        unk_id = self.word2id.get("<UNK>", self.word2id.get("UNK", 1))
        pad_id = self.word2id.get("<PAD>", self.word2id.get("PAD", 0))

        ids = [self.word2id.get(w, unk_id) for w in tokens]
        x = torch.tensor([ids], dtype=torch.long)
        mask = torch.ones_like(x, dtype=torch.long)

        # (batch=1, seq_len)
        return x, mask

    @torch.inference_mode()
    def predict(self, tokens: List[str]) -> List[str]:
        x, mask = self.encode(tokens)
        x = x.to(self.device)
        mask = mask.to(self.device)
        pred_ids = self.model.decode(x, mask)[0]
        tags = [self.id2tag[int(i)] for i in pred_ids]
        return tags

# =========================
# 3) Flask app
# =========================

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CRF_PATH = os.path.join(MODELS_DIR, "CRF_best.joblib")
BILSTM_PATH = os.path.join(MODELS_DIR, "BiLSTM_CRF_best.pt")

predictors: Dict[str, PredictorBase] = {}
# =========================
# 3.5) Model Metrics (from your reports)
# =========================
MODEL_METRICS = {
    "crf": {
        "name": "CRF",
        "valid": {
            "token_f1_all": 0.9904,
            "token_f1_non_o": 0.9046,
            "span_p": 0.9145,
            "span_r": 0.8770,
            "span_f1": 0.8953,
        },
        "test": {
            "token_f1_all": 0.9901,
            "token_f1_non_o": 0.9076,
            "span_p": 0.9474,
            "span_r": 0.8926,
            "span_f1": 0.9191,
        },
    },
    "bilstm_crf": {
        "name": "BiLSTM-CRF",
        "valid": {
            "token_f1_all": 0.9850,
            "token_f1_non_o": 0.8690,
            "span_p": 0.9091,
            "span_r": 0.8202,
            "span_f1": 0.8624,
        },
        "test": {
            "token_f1_all": 0.9843,
            "token_f1_non_o": 0.8642,
            "span_p": 0.9253,
            "span_r": 0.8450,
            "span_f1": 0.8834,
        },
    },
}

def load_predictors() -> None:
    # Load CRF
    if os.path.exists(CRF_PATH):
        try:
            predictors["crf"] = CRFPredictor.load(CRF_PATH)
        except Exception as e:
            predictors["crf"] = None  # type: ignore
            print("[WARN] Failed to load CRF:", e)

    # Load BiLSTM-CRF
    if os.path.exists(BILSTM_PATH):
        try:
            predictors["bilstm_crf"] = BiLSTMCRFPredictor.load(BILSTM_PATH)
        except Exception as e:
            predictors["bilstm_crf"] = None  # type: ignore
            print("[WARN] Failed to load BiLSTM-CRF:", e)

load_predictors()

@app.get("/")
def home():
    available = {k: (v is not None) for k, v in predictors.items()}
    return render_template("index.html", available=available)
@app.get("/api/metrics")
def api_metrics():
    model_key = request.args.get("model", "crf")
    m = MODEL_METRICS.get(model_key)
    if not m:
        return jsonify({"ok": False, "error": "Unknown model key"}), 400
    return jsonify({"ok": True, "model": model_key, "metrics": m})
@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    model_key = data.get("model") or "crf"

    if not text:
        return jsonify({"ok": False, "error": "Input text is empty."}), 400

    pred = predictors.get(model_key)
    if pred is None:
        return jsonify({
            "ok": False,
            "error": f"Model '{model_key}' is not available (load failed). Check server logs + README.",
        }), 400

    tokens = simple_vi_tokenize(text)
    if not tokens:
        return jsonify({"ok": False, "error": "No tokens after tokenization."}), 400

    try:
        tags = pred.predict(tokens)
        spans = group_spans(tokens, tags)
        return jsonify({
            "ok": True,
            "model": model_key,
            "tokens": tokens,
            "tags": tags,
            "spans": spans,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # For local dev
    app.run(host="127.0.0.1", port=5000, debug=True)
