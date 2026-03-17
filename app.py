from __future__ import annotations

import os

from flask import Flask

from user_input import bp as user_input_bp
from video_upload import bp as video_upload_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    app.register_blueprint(user_input_bp)
    app.register_blueprint(video_upload_bp)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
