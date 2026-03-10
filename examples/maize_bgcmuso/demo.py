"""Demo: Generate priors for Biome-BGCMuSo maize parameters (Hollos et al., 2022)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from distribird.agent.pipeline import run_batch
from distribird.config import get_settings
from distribird.export.json_export import export_json
from distribird.export.python_export import export_python
from distribird.export.r_export import export_r
from distribird.models import ParameterInput


async def main():
    params_file = Path(__file__).parent / "parameters.json"
    with open(params_file) as f:
        params_data = json.load(f)

    parameters = [ParameterInput(**p) for p in params_data]
    settings = get_settings()

    print(f"Processing {len(parameters)} parameters...")
    batch = await run_batch(parameters, settings)

    for result in batch.results:
        prior = result.prior
        print(f"\n{'='*60}")
        print(f"Parameter: {result.parameter.name}")
        print(f"  Distribution: {prior.display_name()}")
        print(f"  Confidence: {prior.confidence.value}")
        print(f"  Sources: {prior.n_sources}")
        print(f"  Reason: {prior.reason}")
        if result.warnings:
            for w in result.warnings:
                print(f"  WARNING: {w}")

    # Export
    output_dir = Path(__file__).parent
    (output_dir / "priors.json").write_text(export_json(batch))
    (output_dir / "priors.R").write_text(export_r(batch))
    (output_dir / "priors.py").write_text(export_python(batch))
    print(f"\nExported to {output_dir}/priors.{{json,R,py}}")


if __name__ == "__main__":
    asyncio.run(main())
