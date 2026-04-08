"""
One-off script: ingest ALL candidate records from Postgres into Qdrant.
Loops in batches until no more candidates are returned.

Usage:  python -m scripts.full_ingest
"""

import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.postgres import SessionLocal
from app.services.checkpoint_service import CheckpointService
from app.services.ingest_service import IngestService
from app.services.qdrant_service import QdrantService


BATCH_SIZE = 500  # larger batches to speed things up


async def main() -> None:
    qdrant_service = QdrantService()
    checkpoint_service = CheckpointService()

    total_candidates = 0
    total_points = 0
    batch_num = 0

    # First batch: full_reindex=True resets cursor to None
    full_reindex = True

    while True:
        batch_num += 1
        async with SessionLocal() as session:
            service = IngestService(
                qdrant_service=qdrant_service,
                checkpoint_service=checkpoint_service,
            )
            result = await service.run(
                session=session,
                full_reindex=full_reindex,
                limit=BATCH_SIZE,
            )

        processed = result["candidates_processed"]
        upserted = result["points_upserted"]
        cursor = result["cursor_updated_at"]

        total_candidates += processed
        total_points += upserted

        print(
            f"Batch {batch_num}: {processed} candidates, "
            f"{upserted} points upserted (cursor → {cursor})"
        )

        if processed == 0:
            break

        # Subsequent batches continue from the cursor
        full_reindex = False

    print(f"\nDone! Total candidates: {total_candidates}, total points: {total_points}")


if __name__ == "__main__":
    asyncio.run(main())
