from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Groq / OpenAI Client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/validate", response_class=HTMLResponse)
async def validate(
    request: Request,
    startup_name: str = Form(...),
    pitch: str = Form(...),
    description: str = Form(...),
    target_customer: str = Form(...),
    revenue_model: str = Form(...),
    geography: str = Form(...),
    stage: str = Form(...),
    additional_context: str = Form(None)
):
    prompt = f"""You are a Silicon Valley VC Analyst. 
    Analyze this startup idea:
    Name: {startup_name}
    Pitch: {pitch}
    Description: {description}
    Target: {target_customer}
    Revenue Model: {revenue_model}
    Geography: {geography}
    Stage: {stage}
    Context: {additional_context}

    Return ONLY valid JSON:
    {{
      "market_viability_score": 0-100,
      "strengths": "point1, point2",
      "weaknesses": "point1, point2",
      "opportunities": "point1, point2",
      "threats": "point1, point2",
      "competitors": "list of competitors",
      "improvements": "strategic advice"
    }}
    """

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )

    analysis = json.loads(resp.choices[0].message.content)

    return templates.TemplateResponse(
        "report.html", 
        {
            "request": request, 
            "startup_name": startup_name,
            "pitch": pitch,
            "analysis": analysis
        }
    )