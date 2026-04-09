from fastapi import APIRouter, Depends

from src.core.deps import get_current_user
from src.models.user import User
from src.schemas.user import UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserRead)
async def get_me(user: User = Depends(get_current_user)):
    return user
