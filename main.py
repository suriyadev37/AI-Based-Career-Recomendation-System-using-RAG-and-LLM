from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import os, json, httpx
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from rag import get_rag_context, build_indexes

load_dotenv()

# LOCAL LLM CONFIG (Ollama)
OLLAMA_URL = "https://b8eb-2405-201-e034-b05d-f15d-b69b-82a8-3aae.ngrok-free.app/api/generate"

MODEL = "qwen3.5:9b" 
#MODEL = "qwen2.5:7b-instruct-q4_K_M"

#

# COURSE SEARCH CONFIG (Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL     = "https://api.tavily.com/search"

print(f"[ENV] Using local Ollama LLM: {MODEL} at {OLLAMA_URL}")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# BUILD VECTOR INDEXES AT STARTUP
build_indexes(force=False)

# ─── COURSE SEARCH via Tavily 

async def search_courses(career_title: str) -> list:
    """
    Search for real online courses for the given career title using Tavily.
    Returns a deduplicated list of up to 5 unique courses with valid URLs.
    Falls back to empty list if Tavily key is missing or search fails.
    """
    if not TAVILY_API_KEY:
        print("[COURSES] No TAVILY_API_KEY set — skipping course search.")
        return []

    query = f"best online courses for {career_title} site:coursera.org OR site:udemy.com OR site:edx.org OR site:simplilearn.com OR site:greatlearning.in OR site:fast.ai OR site:freecodecamp.org"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                TAVILY_URL,
                json={
                    "api_key":        TAVILY_API_KEY,
                    "query":          query,
                    "search_depth":   "basic",
                    "max_results":    10,
                    "include_answer": False,
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

        # ── Deduplicate by URL and pick best icon per platform ──────────────
        PLATFORM_ICONS = {
            "coursera":       "🎓",
            "udemy":          "📚",
            "edx":            "🏛️",
            "simplilearn":    "⚡",
            "greatlearning":  "🌟",
            "fast.ai":        "🚀",
            "freecodecamp":   "💻",
            "linkedin":       "💼",
            "pluralsight":    "🎯",
            "skillshare":     "🎨",
            "youtube":        "▶️",
        }

        seen_urls     = set()
        seen_titles   = set()
        courses       = []

        for r in results:
            url   = (r.get("url") or "").strip()
            title = (r.get("title") or "").strip()

            # Skip empty, duplicate URLs, or duplicate titles
            if not url or not title:
                continue
            norm_url   = url.rstrip("/").lower()
            norm_title = title.lower()
            if norm_url in seen_urls or norm_title in seen_titles:
                continue

            # Must look like a real course page (not just a homepage)
            if len(url) < 20:
                continue

            # Detect platform & icon
            icon     = "🎓"
            platform = "Online Course"
            for key, ico in PLATFORM_ICONS.items():
                if key in url.lower():
                    icon     = ico
                    platform = key.replace("greatlearning", "Great Learning").title()
                    break

            seen_urls.add(norm_url)
            seen_titles.add(norm_title)
            courses.append({
                "name":     title,
                "link":     url,
                "platform": platform,
                "icon":     icon,
            })

            if len(courses) >= 5:
                break

        print(f"[COURSES] Found {len(courses)} unique courses for '{career_title}'")
        return courses

    except Exception as e:
        print(f"[COURSES] Search failed: {e}")
        return []


# ─── AI CALL — Async Local Ollama LLM ────────────────────────────────────────

async def call_ai(prompt: str):
    """Optimized async Ollama call for qwen3.5:9b"""

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:

            response = await client.post(
                OLLAMA_URL,
                 json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,

                    # SPEED SETTINGS
                    "temperature": 0.2,
                    "top_p": 0.7,
                    "repeat_penalty": 1.0,

                    # BIGGEST SPEED BOOST
                    "num_predict": 180,

                    # Keep model loaded in memory
                    "keep_alive": "30m",

                    "format": "json",
                    
                    "think": False,

                    # Smaller context = faster
                    "num_ctx": 1024
                }
            )

            response.raise_for_status()

            print("Reponse ==== ", response)

            data = response.json()

            print("\n[OLLAMA RESPONSE]")
            print(data)

            raw = data.get("response", "").strip()

            if not raw:
                return json.dumps({
                    "error": "Empty response from Ollama."
                })

            return raw

    except httpx.ConnectError:
        return json.dumps({
            "error": "Cannot connect to Ollama server."
        })

    except httpx.ReadTimeout:
        return json.dumps({
            "error": "Ollama request timed out."
        })

    except Exception as e:
        return json.dumps({
            "error": f"Ollama error: {str(e)}"
        })

# ─── PROMPT ───────────────────────────────────────────────────────────────────

def build_prompt(user_type: str, data: dict) -> str:
    tasks = {
        "uneducated": "Recommend ONE vocational/trade career requiring NO formal degree. It MUST align perfectly with their stated Interest and Existing Skills.",
        "school":     "Recommend ONE specific higher education path (e.g., B.Tech Computer Science, B.Com, BA, etc.). It MUST logically follow their Favourite Subject, stated Skills, and Career Goal.",
        "graduate":   "Recommend ONE specific job role or higher study path. It MUST be a direct, logical progression from their Degree, Specialization, and existing Projects/Skills.",
        "job_seeker": "Recommend ONE specific next job role. It MUST align directly with their Current Role, Experience, Skills, and Reason for Change.",
    }

    profiles = {
        "uneducated": f"Can Read: {data.get('can_read')}\nCan Write: {data.get('can_write')}\nInterest: {data.get('interest')}\nExisting Skills: {data.get('skills')}\nPhysical Work Preference: {data.get('physical_work_preference')}",
        "school":     f"Qualification: {data.get('qualification')}\nBoard: {data.get('board')}\nSubjects & Marks:\n{data.get('subjects')}\nFavourite Subject: {data.get('fav_subject')}\nExam Preparing: {data.get('exam')}\nCareer Goal: {data.get('career_type')}\nExtra Curricular: {data.get('extra')}\nSkills: {data.get('skills')}\nHobbies: {data.get('hobbies')}",
        "graduate":   f"Degree: {data.get('degree')}\nSpecialization: {data.get('specialization')}\nCGPA: {data.get('cgpa')}\nCareer Goal: {data.get('career_type')}\nProjects: {data.get('projects')}\nInternships: {data.get('internships')}\nCertifications: {data.get('certs')}\nSkills: {data.get('skills')}\nHobbies: {data.get('hobbies')}",
        "job_seeker": f"Current Role: {data.get('role')}\nExperience: {data.get('experience')} years\nCompany: {data.get('company')}\nCurrent CTC: {data.get('current_ctc')}\nExpected CTC: {data.get('expected_ctc')}\nWork Mode: {data.get('work_mode')}\nProjects: {data.get('projects')}\nAchievements: {data.get('achievements')}\nSkills: {data.get('skills')}\nReason for Change: {data.get('reason')}",
    }

    rag_context = get_rag_context(user_type, data)
    rag_section = (
        f"""REFERENCE DATA:\n{rag_context}\n(Note: Only use the reference data if it logically aligns with the user's explicitly stated skills and goals.)\n"""
        if rag_context else ""
    )

    return f"""You are an expert career advisor. Output STRICTLY in JSON format.

CRITICAL RULES:
1. STRICT ALIGNMENT: The recommendation MUST logically match the user's stated goals, favorite subjects, and skills. (e.g., If the user likes Computer Science, do NOT recommend Pharmacy or Biology).
2. REALISM: Base the roadmap and courses on real-world industry standards.
3. SKILLS: List ALL skills required for this career — both core technical skills and soft skills. Provide at least 10-15 skills. Do NOT limit to 5.
4. ROADMAP: Each roadmap step must have a clear title AND a detailed 'desc' field describing what to learn/do in that phase (2-3 sentences minimum).

{rag_section}
TASK: {tasks.get(user_type, 'Recommend a career.')}

USER PROFILE:
{profiles.get(user_type, 'Unknown.')}

EXPECTED JSON OUTPUT FORMAT:
{{
  "analysis": "Briefly state the user's main interest/skill and logically deduce the best career path in 1-2 sentences. DO NOT skip this step.",
  "title": "[Short clean job title only — e.g. 'QA Engineer', 'Front-End Developer', 'Data Analyst'. DO NOT add industry sector, department, government/private prefix, or any extra words. Max 4 words.]",
  "why": [
    "[Specific reason linking recommendation to user's profile data]",
    "[Specific reason linking recommendation to user's profile data]",
    "[Specific reason linking recommendation to user's profile data]"
  ],
  "skills": [
    "[Skill 1]", "[Skill 2]", "[Skill 3]", "[Skill 4]", "[Skill 5]",
    "[Skill 6]", "[Skill 7]", "[Skill 8]", "[Skill 9]", "[Skill 10]",
    "[Skill 11]", "[Skill 12]", "[Skill 13]", "[Skill 14]", "[Skill 15]"
  ],
  "roadmap": [
    {{"title": "Step 1: [Short Title]", "desc": "[Detailed description of step]"}},
    {{"title": "Step 2: [Short Title]", "desc": "[Detailed description of step]"}},
    {{"title": "Step 3: [Short Title]", "desc": "[Detailed description of step]"}}
  ]
}}"""
    # NOTE: "courses" intentionally removed from LLM prompt —
    # real course URLs are fetched separately via web search (search_courses).


# ─── PARSER ───────────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    """Parses the JSON response safely with fallback defaults."""
    result = {
        "title": "Career Recommendation",
        "why": [], "skills": [], "roadmap": [], "courses": [],
        "raw": text, "error": "",
    }

    if not text:
        result["error"] = "no_response"
        return result

    try:
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        parsed_data = json.loads(clean_text.strip())

        # Only treat as error if it has ONLY an error key (not a valid career response)
        if "error" in parsed_data and "title" not in parsed_data:
            result["error"] = parsed_data["error"]
            return result

        # Update but preserve courses key — real courses are fetched later via Tavily
        llm_courses = parsed_data.pop("courses", None)  # strip LLM-hallucinated courses
        result.update(parsed_data)
        # Restore courses placeholder (will be replaced by Tavily in run_predict)
        result["courses"] = []

        # ── Clean up title: remove sector/industry/govt noise the LLM adds ──
        result["title"] = _clean_title(result.get("title", "Career Recommendation"))

        analysis = parsed_data.get("analysis", "No analysis provided.")
        print(f"[AI LOGIC]: {analysis}")

    except json.JSONDecodeError:
        result["error"] = "Failed to parse AI response into JSON format."

    print(f"[PARSER] '{result['title']}' why={len(result['why'])} skills={len(result['skills'])}")
    return result


def _clean_title(title: str) -> str:
    """
    Strips unwanted industry/sector/govt prefixes and suffixes from career titles.
    e.g. 'Telecommunication QA Engineer' → 'QA Engineer'
         'Government Front-End Web Developer' → 'Front-End Developer'
         'Senior Software Engineer (IT Sector)' → 'Software Engineer'
    """
    import re

    # Common noisy prefix words to remove (case-insensitive)
    # NOTE: 'finance' and 'medical' intentionally excluded — they can be valid title words
    NOISE_PREFIXES = [
        "telecommunication", "telecommunications",
        "government", "govt",
        "private sector", "public sector",
        "information technology", "it sector",
        "telecom", "banking sector",
        "manufacturing", "construction", "retail",
        "e-commerce", "corporate",
        "junior", "senior", "lead", "associate",
        "entry-level", "entry level",
        "mid-level", "mid level",
        "principal", "staff",
    ]

    # Suffixes / parenthetical noise to strip
    NOISE_SUFFIXES = [
        r"\s*\(.*?\)",          # anything in parentheses e.g. (IT Sector)
        r"\s*-\s*(it|tech|telecom|govt|gov|banking|finance|healthcare)\s*$",
        r"\s+sector$",
        r"\s+industry$",
        r"\s+department$",
        r"\s+field$",
    ]

    cleaned = title.strip()

    # Remove parenthetical noise first
    for pattern in NOISE_SUFFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Remove leading noise prefix words one by one
    changed = True
    while changed:
        changed = False
        lower = cleaned.lower()
        for prefix in NOISE_PREFIXES:
            if lower.startswith(prefix + " ") or lower.startswith(prefix + "-"):
                cleaned = cleaned[len(prefix):].strip().lstrip("-").strip()
                lower   = cleaned.lower()
                changed = True

    # Title-case the result cleanly
    cleaned = cleaned.strip(" ,-")
    return cleaned if cleaned else title.strip()


# ─── HELPER: full predict pipeline ───────────────────────────────────────────

async def run_predict(request: Request, user_type: str, data: dict):
    """Shared pipeline: call LLM → parse → fetch real courses → render."""

    prompt = build_prompt(user_type, data)

    raw_response = await call_ai(prompt)

    parsed = parse_response(raw_response)

    # Replace LLM-hallucinated courses with real web-searched ones
    if not parsed.get("error"):
        real_courses = await search_courses(parsed.get("title", ""))
        parsed["courses"] = real_courses
    else:
        parsed["courses"] = []

    print(
        f"[RESULT] "
        f"title={parsed.get('title')} "
        f"why={len(parsed.get('why', []))} "
        f"skills={len(parsed.get('skills', []))} "
        f"roadmap={len(parsed.get('roadmap', []))} "
        f"courses={len(parsed.get('courses', []))}"
    )

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "data": parsed
        }
    )


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/index.html", response_class=HTMLResponse)
async def index_redirect(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/selection.html", response_class=HTMLResponse)
async def selection_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="selection.html",
        context={}
    )


@app.get("/uneducated", response_class=HTMLResponse)
async def uneducated_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="uneducated.html",
        context={}
    )


@app.get("/school", response_class=HTMLResponse)
async def school_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="school.html",
        context={}
    )


@app.get("/graduate", response_class=HTMLResponse)
async def graduate_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="graduate.html",
        context={}
    )


@app.get("/jobseeker", response_class=HTMLResponse)
async def jobseeker_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="jobseeker.html",
        context={}
    )


@app.get("/debug-test", response_class=HTMLResponse)
async def debug_test(request: Request):

    raw = await call_ai(
        build_prompt(
            "uneducated",
            {
                "can_read": "Yes",
                "can_write": "Yes",
                "interest": "Cooking",
                "skills": "Basic cooking",
                "physical_work_preference": "Mix of both",
            }
        )
    )

    return templates.TemplateResponse(
        request=request,
        name="debug.html",
        context={
            "raw": raw,
            "parsed": parse_response(raw),
        }
    )

@app.post("/predict-uneducated", response_class=HTMLResponse)
async def predict_uneducated(
    request: Request,
    can_read: str = Form(...), can_write: str = Form(...),
    interest: str = Form(...), skills: str = Form(...),
    physical_work_preference: str = Form(...),
):
    return await run_predict(request, "uneducated", {
        "can_read": can_read, "can_write": can_write,
        "interest": interest, "skills": skills,
        "physical_work_preference": physical_work_preference,
    })


@app.post("/predict-school", response_class=HTMLResponse)
async def predict_school(
    request: Request,
    qualification: str = Form(...), board_of_study: str = Form(...),
    subjects: List[str] = Form(...), marks: List[str] = Form(...),
    fav_subject: str = Form(...), exam_name: Optional[str] = Form(None),
    career_type: str = Form(...), extra_curricular: str = Form(...),
    skills: List[str] = Form(...), proficiencies: List[str] = Form(...),
    hobbies: str = Form(...),
):
    if len(subjects) != len(marks):
        raise HTTPException(status_code=400, detail="Mismatch between number of subjects and marks.")
    if len(skills) != len(proficiencies):
        raise HTTPException(status_code=400, detail="Mismatch between number of skills and proficiencies.")

    return await run_predict(request, "school", {
        "qualification": qualification, "board": board_of_study,
        "subjects": "\n".join([f"{s}: {m}%" for s, m in zip(subjects, marks)]),
        "fav_subject": fav_subject, "exam": exam_name or "None",
        "career_type": career_type, "extra": extra_curricular,
        "skills": "\n".join([f"{s} ({p})" for s, p in zip(skills, proficiencies)]),
        "hobbies": hobbies,
    })


@app.post("/predict-graduate", response_class=HTMLResponse)
async def predict_graduate(
    request: Request,
    degree_name: str = Form(...), specialization: str = Form(...),
    cgpa: str = Form(...), career_type: str = Form(...),
    projects: Optional[List[str]] = Form(None),
    internships: Optional[List[str]] = Form(None),
    certs: Optional[List[str]] = Form(None),
    skills: List[str] = Form(...), skill_levels: List[str] = Form(...),
    hobbies: str = Form(...),
):
    if len(skills) != len(skill_levels):
        raise HTTPException(status_code=400, detail="Mismatch between number of skills and skill levels.")

    return await run_predict(request, "graduate", {
        "degree": degree_name, "specialization": specialization,
        "cgpa": cgpa, "career_type": career_type,
        "projects": "\n".join(projects or []) or "None",
        "internships": "\n".join(internships or []) or "None",
        "certs": "\n".join(certs or []) or "None",
        "skills": "\n".join([f"{s} ({l})" for s, l in zip(skills, skill_levels)]),
        "hobbies": hobbies,
    })


@app.post("/predict-jobseeker", response_class=HTMLResponse)
async def predict_jobseeker(
    request: Request,
    current_role: str = Form(...), experience_years: str = Form(...),
    current_company: str = Form(...),
    skills: List[str] = Form(...), proficiencies: List[str] = Form(...),
    current_ctc: str = Form(...), expected_ctc: str = Form(...),
    work_mode: str = Form(...),
    projects: Optional[List[str]] = Form(None),
    achievements: Optional[List[str]] = Form(None),
    change_reason: str = Form(...),
):
    if len(skills) != len(proficiencies):
        raise HTTPException(status_code=400, detail="Mismatch between number of skills and proficiencies.")

    return await run_predict(request, "job_seeker", {
        "role": current_role, "experience": experience_years,
        "company": current_company, "current_ctc": current_ctc,
        "expected_ctc": expected_ctc, "work_mode": work_mode,
        "projects": "\n".join(projects or []) or "None",
        "achievements": "\n".join(achievements or []) or "None",
        "skills": "\n".join([f"{s} ({p})" for s, p in zip(skills, proficiencies)]),
        "reason": change_reason,
    })
