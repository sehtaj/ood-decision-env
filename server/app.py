# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the operational risk triage environment."""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from exc

try:
    from ..models import TriageAction, TriageObservation
    from .my_env_environment import MyEnvironment
except ImportError:
    from models import TriageAction, TriageObservation  # type: ignore
    from server.my_env_environment import MyEnvironment  # type: ignore


app = create_app(
    MyEnvironment,
    TriageAction,
    TriageObservation,
    env_name="operational_risk_triage",
    max_concurrent_envs=4,
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

def main() -> None:
    """Run the OpenEnv-compatible FastAPI server locally."""

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
