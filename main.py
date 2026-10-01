
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from qna import answer_question
from explanation_module import explain_concept
from quiz_module import generate_quiz
from summary_module import summarize_text
from learning_path import get_learning_recommendations

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="EduGenie",
    description="Google Gemini Powered Learning Assistant",
    version="1.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30000)
    level: str = "beginner"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai_configured": bool(os.getenv("GEMINI_API_KEY"))
    }


async def run_task(function, *args):
    try:
        result = await function(*args)
        return {"result": result}
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc)}
        )


@app.post("/qa")
async def qa(payload: QuestionRequest):
    return await run_task(
        answer_question,
        payload.question
    )


@app.post("/explain")
async def explain(payload: TextRequest):
    return await run_task(
        explain_concept,
        payload.text,
        payload.level
    )


@app.post("/quiz")
async def quiz(payload: TextRequest):
    return await run_task(
        generate_quiz,
        payload.text
    )


@app.post("/summarize")
async def summarize(payload: TextRequest):
    return await run_task(
        summarize_text,
        payload.text
    )


@app.post("/learn/recommendations")
async def recommendations(payload: TextRequest):
    return await run_task(
        get_learning_recommendations,
        payload.text,
        payload.level
    )


@app.post("/api/{task}")
async def api_task(task: str, payload: TextRequest):
    handlers = {
        "qa": lambda: answer_question(payload.text),
        "explain": lambda: explain_concept(
            payload.text, payload.level
        ),
        "quiz": lambda: generate_quiz(payload.text),
        "summarize": lambda: summarize_text(payload.text),
        "recommend": lambda: get_learning_recommendations(
            payload.text, payload.level
        )
    }

    if task not in handlers:
        return JSONResponse(
            status_code=404,
            content={"detail": "Unknown task"}
        )

    return await run_task(handlers[task])