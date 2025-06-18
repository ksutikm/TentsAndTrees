from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import List
import numpy as np
import random

from src.grid_optim import GridOptim

# Импортируй GridOptim из твоего файла, если он в другом модуле
# from your_module import GridOptim

router = APIRouter()

# ======= Входная модель =======
class SizeRequest(BaseModel):
    size: int

# ======= Выходная модель =======
class GridResponse(BaseModel):
    grid: List[List[int]]
    row: List[int]
    col: List[int]

# ======= FastAPI endpoint =======
@router.post("/generate", response_model=GridResponse)
async def generate_grid(request: SizeRequest):
    grid_obj = GridOptim(request.size)
    return GridResponse(
        grid=grid_obj.grid.tolist(),
        row=grid_obj.row_constraints,
        col=grid_obj.col_constraints
    )
