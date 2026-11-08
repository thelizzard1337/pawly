import base64
import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import tuple_

from app.api.errors import fail


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None


def encode_cursor(row) -> str:
    return (
        base64.urlsafe_b64encode(
            json.dumps(
                [row.created_at.isoformat(), str(row.id)],
            ).encode()
        )
        .decode()
        .rstrip("=")
    )


def decode_cursor(value: str):
    try:
        if len(value) > 300:
            raise ValueError
        stamp, uid = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
        if not isinstance(stamp, str) or not isinstance(uid, str):
            raise ValueError
        dt = datetime.fromisoformat(stamp)
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError
        return dt, UUID(uid)
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        from fastapi import HTTPException

        raise HTTPException(422, detail={"code": "invalid_cursor"}) from exc


def paginate(statement, model, cursor: str | None, limit: int):
    if not 1 <= limit <= 100:
        fail(422, "invalid_limit")
    if cursor:
        dt, uid = decode_cursor(cursor)
        statement = statement.where(tuple_(model.created_at, model.id) < tuple_(dt, uid))
    return statement.order_by(model.created_at.desc(), model.id.desc()).limit(limit + 1)


def page_parts(rows, limit):
    return rows[:limit], encode_cursor(rows[limit - 1]) if len(rows) > limit else None
