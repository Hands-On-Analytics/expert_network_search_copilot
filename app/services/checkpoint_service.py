from dataclasses import dataclass
from uuid import UUID


@dataclass
class Checkpoint:
    cursor_candidate_id: UUID | None = None


class CheckpointService:
    def __init__(self) -> None:
        self._checkpoint = Checkpoint(cursor_candidate_id=None)

    async def get_cursor(self) -> UUID | None:
        return self._checkpoint.cursor_candidate_id

    async def set_cursor(self, value: UUID | None) -> None:
        self._checkpoint.cursor_candidate_id = value
