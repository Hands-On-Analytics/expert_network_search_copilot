from __future__ import annotations

import json

import typer
import uvicorn

from candidate_ingest_search.ingestion import CandidateIngestionPipeline
from candidate_ingest_search.settings import get_settings

cli = typer.Typer(help="Candidate vector ingestion commands.")


@cli.command("ingest")
def ingest_command(
    full_refresh: bool = typer.Option(
        True,
        "--full-refresh/--no-full-refresh",
        help="Reset the vector collection before ingesting.",
    ),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Optional candidate limit for smoke runs.",
    ),
) -> None:
    settings = get_settings()
    pipeline = CandidateIngestionPipeline.from_settings(settings)
    result = pipeline.ingest(full_refresh=full_refresh, limit=limit)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@cli.command("serve")
def serve_command(
    host: str = typer.Option("0.0.0.0", help="Host interface for uvicorn."),
    port: int = typer.Option(8000, help="Port for uvicorn."),
    reload: bool = typer.Option(False, help="Enable autoreload."),
) -> None:
    uvicorn.run("candidate_ingest_search.api:app", host=host, port=port, reload=reload)


def main() -> None:
    cli()
