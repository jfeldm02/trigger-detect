from __future__ import annotations

import os
from typing import List

from flask import Blueprint, jsonify, render_template, request, session
from huggingface_hub import InferenceClient
import ast

bp = Blueprint("user_input", __name__)

HF_MODEL = os.getenv("HF_MODEL", "google/gemma-2-2b-it")
client = InferenceClient(token=os.getenv("HF_TOKEN"))


def _safe_parse_tags(text: str) -> List[str]:
    """
    Parse the model output into a list of tags.
    Expects a JSON array of strings. Falls back to best-effort extraction.
    """
    data = None
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        list_data = text[start : end + 1]
        data = ast.literal_eval(list_data)
    if data is not None:
        return data
    else:
        return []


@bp.post("/extract-tags")
def extract_tags():
    request_body = request.get_json(silent=True) or {}
    user_text = request_body.get("text", "").strip()
    max_tags = int(request_body.get("max_tags"))

    if not os.getenv("HF_TOKEN"):
        return jsonify({"error": "HF_TOKEN secret is not set"}), 500

    system_prompt = (
        "You must extract tags from a given text. The tags can be direct entities from the text or example options from the text. "
        "The tag should always be a tangible object. The more you generalize the better. "
        "You must provide these tags in a Python list format i.e. between two square brackets ([]). "
        "Any text that discusses, praises or degrades sensitive topics such as Gender, Caste, Creed, Religion, Political Inclination, "
        "Nationality, Citizenship Status, Immigration Status, Countries and their flags or Sex must not be encouraged. "
        "In such a case you must always return an empty list ([]) "
        f"The tags must be relevant to the text provided. Extract at most {max_tags} tags. An example is provided for your reference: \n"
        "EXAMPLE: \n "
        "INPUT text: I am afraid of creepy crawlies or anything with more than 4 legs, especially spiders. They freak me out."
        "Output: ['insects', 'creepy crawlies', 'spiders', 'centipede', 'tarantula']"
    )

    prompt = f"<start_of_turn>user\n{system_prompt}\n\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

    try:
        output_text = client.text_generation(
            prompt,
            model=HF_MODEL,
            max_new_tokens=512,
        )
    except Exception as exc:
        return jsonify({"error": f"HF Inference API error: {exc}"}), 502
    tags = _safe_parse_tags(output_text)

    return jsonify(
        {
            "tags": tags,
            "model": HF_MODEL,
            "message": "Ethics Violation" if len(tags) == 0 else "Tags Extracted",
        }
    )


@bp.patch("/edit-tags")
def edit_tags():
    request_body = request.get_json(silent=True) or {}
    tags = request_body.get("tags")

    if not isinstance(tags, list):
        return jsonify({"error": "Missing or invalid JSON field: tags"}), 400

    cleaned_tags: List[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            return jsonify({"error": "All tags must be strings"}), 400
        cleaned = tag.strip()
        if cleaned:
            cleaned_tags.append(cleaned)

    session["tags"] = cleaned_tags
    return jsonify({"tags": cleaned_tags})


@bp.get("/")
def index():
    return render_template("index.html")
