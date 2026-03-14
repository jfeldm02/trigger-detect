from __future__ import annotations

import json
import os
from typing import List

from flask import Blueprint, jsonify, render_template, request, session
from openai import OpenAI
import ast

bp = Blueprint("user_input", __name__)
client = OpenAI()


def _safe_parse_tags(text: str) -> List[str]:
    """
    Parse the model output into a list of tags.
    Expects a JSON array of strings. Falls back to best-effort extraction.
    """
    # Best-effort extraction if the model wrapped the JSON in extra text.
    data = None
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        list_data = text[start:end +1]
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

    output_text = ""
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5"),
        input=[
            {
                "role": "system",
                "content": (
                    "You must extract tags from a given text. The tags can be direct entities from the text or example options from the text. The tag should always be a tangible object. The more you generalize the better.You must provide these tags in a Python list format i.e. between two square brackets ([])."+
                    f"The tags must be relevant to the text provided. Extract at most {max_tags} tags. An example is provided for your reference: \n"+
                    "EXAMPLE: \n "+
                    "INPUT text: I am afraid of creepy crawlies or anything with more than 4 legs, especially spiders. They freak me out."+
                    "Output: ['insects', 'creepy crawlies', 'spiders', 'centipede', 'tarantula']"
                ),
            },
            {"role": "user", "content": user_text},
        ],
    )

    data = json.loads(response.json())
    outputs = data['output']
    content = None
    for output in outputs:
        if 'content' in output.keys() and output.get("content") is not None:
            content = output['content'][0]
    if content is not None:
        if 'text' in content.keys() and content['text'] is not None:
            output_text = content['text']
    
    tags = _safe_parse_tags(output_text or "")

    return jsonify(
        {
            "tags": tags,
            "model": response.model,
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
