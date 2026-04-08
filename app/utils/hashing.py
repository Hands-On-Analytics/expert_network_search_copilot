import hashlib
import orjson


def stable_hash(payload: dict) -> str:
    raw = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()