#!/bin/env/python
"""Script to run the benchmark on a vector database."""

from pathlib import Path
from typing import Annotated

import typer

from inatinqperf.benchmark.benchmark import Benchmarker

app = typer.Typer()


@app.command("run")
def main(
    config_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="The configuration file to use for running the benchmark.",
        ),
    ],
    base_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="The base path relative to which various artifacts are saved.",
        ),
    ] = Path(__file__).parent.parent,
) -> None:
    """Vector Database agnostic benchmark."""

    benchmarker = Benchmarker(config_file, base_path=base_path)

    benchmarker.run()


if __name__ == "__main__":
    app()
