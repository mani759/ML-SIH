"""
Thin FastAPI wrapper around get_risk_score.py / risk_model.py.
Deployed on Render's free tier as an independent Python service.
AI Studio's generated Node.js backend calls this over HTTP -- it
does not (and cannot) run Python itself.
"""

import os
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from get_risk_score import get_risk_score
import risk_model

app = FastAPI(title="MPLAD Risk Scoring Service")

_DIR = os.path.dirname(os.path.abspath(__file__))
_util = joblib.load(os.path.join(_DIR, "utilization_benchmark.pkl"))


class ProjectIn(BaseModel):
    project_id: str
    state: str
    work_category: str
    mp_name: str
    sanctioned_amount: float
    actual_expenditure: float
    start_date: str
    expected_completion: str
    actual_completion: Optional[str] = None
    status: str
    has_tender_on_file: bool
    has_mp_recommendation: bool


@app.get("/")
def health():
    # Render (and you, warming it up before a demo) can hit this
    # to confirm the service is awake and responding.
    return {"status": "ok", "service": "mplad-risk-scoring"}


@app.post("/score")
def score(project: ProjectIn):
    try:
        return get_risk_score(project.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/duplicates")
def duplicates(projects: List[ProjectIn]):
    if not projects:
        return {"duplicate_flags": []}
    df = pd.DataFrame([p.dict() for p in projects])
    try:
        flags = risk_model.find_duplicates(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"duplicate_flags": flags}


@app.get("/utilization-benchmark/{state}")
def utilization_benchmark(state: str):
    state_util = _util["state_utilization"].get(state, _util["national_avg"])
    return {
        "state": state,
        "state_utilization": round(float(state_util), 3),
        "national_avg": round(float(_util["national_avg"]), 3),
    }
