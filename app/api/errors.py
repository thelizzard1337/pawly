from fastapi import HTTPException


def fail(status: int, code: str):
    raise HTTPException(status, detail={"code": code})
