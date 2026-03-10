"""FastAPI endpoints for Distribird."""

from __future__ import annotations

import hmac

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from distribird.agent.pipeline import run_batch, run_parameter
from distribird.config import get_settings
from distribird.export.json_export import export_json, result_to_dict
from distribird.export.python_export import export_python
from distribird.export.r_export import export_r
from distribird.models import ParameterInput

app = FastAPI(
    title="Distribird API",
    description="Literature-informed Prior distributions for Bayesian model calibration",
    version="0.1.0",
)

security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    settings = get_settings()
    username_ok = hmac.compare_digest(
        credentials.username.encode(), settings.auth_username.encode()
    )
    password_ok = hmac.compare_digest(
        credentials.password.encode(), settings.auth_password.encode()
    )
    if not (username_ok and password_ok):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials.username


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/parameter")
async def process_parameter(
    parameter: ParameterInput, _user: str = Depends(verify_credentials)
) -> dict[str, object]:
    """Process a single parameter and return its fitted prior."""
    settings = get_settings()
    try:
        result = await run_parameter(parameter, settings)
        return result_to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch")
async def process_batch(
    parameters: list[ParameterInput], _user: str = Depends(verify_credentials)
) -> dict[str, object]:
    """Process multiple parameters and return fitted priors."""
    settings = get_settings()
    try:
        batch = await run_batch(parameters, settings)
        return {"results": [result_to_dict(r) for r in batch.results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/export/json")
async def export_json_endpoint(
    parameters: list[ParameterInput], _user: str = Depends(verify_credentials)
) -> dict[str, str]:
    """Process parameters and export as JSON."""
    settings = get_settings()
    batch = await run_batch(parameters, settings)
    return {"export": export_json(batch)}


@app.post("/api/v1/export/r")
async def export_r_endpoint(
    parameters: list[ParameterInput], _user: str = Depends(verify_credentials)
) -> dict[str, str]:
    """Process parameters and export as R script."""
    settings = get_settings()
    batch = await run_batch(parameters, settings)
    return {"export": export_r(batch)}


@app.post("/api/v1/export/python")
async def export_python_endpoint(
    parameters: list[ParameterInput], _user: str = Depends(verify_credentials)
) -> dict[str, str]:
    """Process parameters and export as Python script."""
    settings = get_settings()
    batch = await run_batch(parameters, settings)
    return {"export": export_python(batch)}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
