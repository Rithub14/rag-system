from uuid import uuid4

from fastapi import APIRouter

router = APIRouter()


@router.get("/session")
def new_session() -> dict:
    return {"user_id": str(uuid4())}
