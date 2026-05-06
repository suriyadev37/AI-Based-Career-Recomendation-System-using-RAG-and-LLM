import json
import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions

# PATH SETUP

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# EMBEDDING MODEL
# Uses sentence-transformers all-MiniLM-L6-v2
# ~80MB, runs fully on CPU, no GPU needed

EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# CHROMA CLIENT (persistent - survives restarts)

_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Three separate collections - one per user type
_col_vocational = _client.get_or_create_collection(
    name="vocational", embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"}
)
_col_school = _client.get_or_create_collection(
    name="school", embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"}
)
_col_graduate = _client.get_or_create_collection(
    name="graduate", embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"}
)

# UTILITY

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[RAG] WARNING: {filename} not found at {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"[RAG] ERROR: Failed to parse {filename}: {e}")
        return []

def format_list(items):
    if not items:
        return "Not specified"
    if isinstance(items, list):
        return ", ".join(str(i) for i in items)
    return str(items)

def stable_id(collection_name: str, index: int, text: str) -> str:
    """Generate a guaranteed-unique ID using collection name + index + content hash."""
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{collection_name}_{index}_{content_hash}"

# DOCUMENT BUILDERS
# Each converts a JSON record into a rich
# searchable text string + metadata dict

def build_vocational_doc(job: dict):
    doc = (
        f"Job Title: {job.get('job_title', '')}\n"
        f"Description: {job.get('description', '')}\n"
        f"Interest Categories: {format_list(job.get('interest_category', []))}\n"
        f"Required Skills: {format_list(job.get('required_skills', []))}\n"
        f"Physical Work Level: {job.get('physical_work_level', '')}\n"
        f"Learning Path: {format_list(job.get('learning_path', []))}\n"
        f"Average Salary: {job.get('average_salary', '')}\n"
        f"Recommended Courses: {format_list(job.get('recommended_courses', []))}"
    )
    meta = {
        "job_title":           str(job.get("job_title", "")),
        "physical_work_level": str(job.get("physical_work_level", "")),
        "average_salary":      str(job.get("average_salary", "")),
    }
    return doc, meta

def build_school_doc(course: dict):
    doc = (
        f"Course: {course.get('course_name', '')}\n"
        f"Stream: {course.get('recommended_for_stream', '')}\n"
        f"Duration: {course.get('duration', '')}\n"
        f"Required Skills: {format_list(course.get('required_skills', []))}\n"
        f"Career Outcomes: {format_list(course.get('career_outcomes', []))}\n"
        f"Top Colleges: {format_list(course.get('top_colleges', []))}\n"
        f"Entrance Exams: {format_list(course.get('entrance_exams', []))}\n"
        f"Average Salary: {course.get('average_salary', '')}"
    )
    meta = {
        "course_name":            str(course.get("course_name", "")),
        "recommended_for_stream": str(course.get("recommended_for_stream", "")),
        "average_salary":         str(course.get("average_salary", "")),
    }
    return doc, meta

def build_graduate_doc(career: dict):
    doc = (
        f"Career: {career.get('career', '')}\n"
        f"Industry: {career.get('industry', '')}\n"
        f"Required Degree: {career.get('required_degree', '')}\n"
        f"Required Skills: {format_list(career.get('required_skills', []))}\n"
        f"Average Salary: {career.get('average_salary', '')}\n"
        f"Job Outlook: {career.get('job_outlook', '')}\n"
        f"Learning Roadmap: {format_list(career.get('learning_roadmap', []))}\n"
        f"Top Companies: {format_list(career.get('top_companies', []))}\n"
        f"Recommended Certifications: {format_list(career.get('recommended_certifications', []))}"
    )
    meta = {
        "career":          str(career.get("career", "")),
        "industry":        str(career.get("industry", "")),
        "required_degree": str(career.get("required_degree", "")),
        "average_salary":  str(career.get("average_salary", "")),
        "job_outlook":     str(career.get("job_outlook", "")),
    }
    return doc, meta

# INDEX BUILDER
# Loads JSON -> embeds -> stores in ChromaDB
# Skips automatically if already indexed

def _index_collection(collection, records: list, build_fn, label: str):
    """Batch-embed and insert all records into a ChromaDB collection."""
    if collection.count() == len(records):
        print(f"[RAG] '{label}' already indexed ({len(records)} docs) — skipping.")
        return

    print(f"[RAG] Indexing '{label}' — {len(records)} records...")

    # Clear existing if partial
    if collection.count() > 0:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    BATCH = 100  # embed 100 at a time to avoid memory spikes
    for i in range(0, len(records), BATCH):
        batch = records[i:i + BATCH]
        docs, metas, ids = [], [], []
        for j, rec in enumerate(batch):
            doc_text, meta = build_fn(rec)
            global_index = i + j
            doc_id = stable_id(label, global_index, doc_text)
            docs.append(doc_text)
            metas.append(meta)
            ids.append(doc_id)

        collection.upsert(documents=docs, metadatas=metas, ids=ids)
        print(f"[RAG]   '{label}' — {min(i + BATCH, len(records))}/{len(records)} indexed")

    print(f"[RAG] '{label}' indexing complete — {collection.count()} vectors stored.")


def build_indexes(force: bool = False):
    """
    Build all three vector indexes.
    Called once at startup in main.py.
    Safe to call every restart — skips if already indexed.
    Pass force=True to rebuild from scratch (e.g. after updating JSON files).
    """
    global _col_vocational, _col_school, _col_graduate

    if force:
        print("[RAG] Force rebuild — dropping existing collections...")
        for name in ["vocational", "school", "graduate"]:
            try:
                _client.delete_collection(name)
            except Exception:
                pass
        _col_vocational = _client.get_or_create_collection(
            "vocational", embedding_function=EMBED_FN,
            metadata={"hnsw:space": "cosine"})
        _col_school = _client.get_or_create_collection(
            "school", embedding_function=EMBED_FN,
            metadata={"hnsw:space": "cosine"})
        _col_graduate = _client.get_or_create_collection(
            "graduate", embedding_function=EMBED_FN,
            metadata={"hnsw:space": "cosine"})

    vocational_data = load_json("vocational_jobs_500_rag_ready.json")
    school_data     = load_json("school_course_rag_500.json")
    graduate_data   = load_json("career_db_1000_realistic.json")

    _index_collection(_col_vocational, vocational_data, build_vocational_doc, "vocational")
    _index_collection(_col_school,     school_data,     build_school_doc,     "school")
    _index_collection(_col_graduate,   graduate_data,   build_graduate_doc,   "graduate")

    print(
        f"[RAG] All indexes ready — "
        f"Vocational: {_col_vocational.count()}, "
        f"School: {_col_school.count()}, "
        f"Graduate: {_col_graduate.count()}"
    )

# QUERY BUILDERS
# Rich natural-language queries for semantic search.
# The more descriptive the query, the better the match.

def _query_uneducated(data: dict) -> str:
    return (
        f"vocational trade job for someone interested in {data.get('interest', '')} "
        f"with existing skills in {data.get('skills', '')} "
        f"physical work preference is {data.get('physical_work_preference', '')} "
        f"no formal degree required practical hands-on career"
    )

def _query_school(data: dict) -> str:
    return (
        f"higher education course for student who loves {data.get('fav_subject', '')} "
        f"wants career in {data.get('career_type', '')} "
        f"has skills in {data.get('skills', '')} "
        f"hobbies are {data.get('hobbies', '')} "
        f"completed {data.get('qualification', '')} from {data.get('board', '')} board"
    )

def _query_graduate(data: dict) -> str:
    return (
        f"career path for {data.get('degree', '')} graduate "
        f"specialization in {data.get('specialization', '')} "
        f"CGPA {data.get('cgpa', '')} "
        f"skills include {data.get('skills', '')} "
        f"career goal is {data.get('career_type', '')} "
        f"has done projects in {data.get('projects', '')} "
        f"internships at {data.get('internships', '')}"
    )

def _query_job_seeker(data: dict) -> str:
    return (
        f"next career move for {data.get('role', '')} "
        f"with {data.get('experience', '')} years of experience "
        f"skills are {data.get('skills', '')} "
        f"expecting salary {data.get('expected_ctc', '')} "
        f"reason for change {data.get('reason', '')} "
        f"prefers {data.get('work_mode', '')} work"
    )

# SEMANTIC SEARCH

def _semantic_search(collection, query: str, n_results: int = 3) -> list:
    """
    Query ChromaDB using semantic similarity (cosine distance).
    Returns list of matching document strings.
    """
    try:
        count = collection.count()
        if count == 0:
            print(f"[RAG] WARNING: Collection '{collection.name}' is empty.")
            print(f"[RAG] Did you call build_indexes() at startup in main.py?")
            return []

        actual_n = min(n_results, count)
        results = collection.query(
            query_texts=[query],
            n_results=actual_n,
            include=["documents", "distances", "metadatas"]
        )

        docs      = results["documents"][0]
        distances = results["distances"][0]
        metas     = results["metadatas"][0]

        # Log similarity scores for debugging
        for i, (dist, meta) in enumerate(zip(distances, metas)):
            similarity = round(1 - dist, 3)
            label = list(meta.values())[0] if meta else f"result_{i+1}"
            print(f"[RAG]   rank {i+1}: '{label}' — similarity={similarity}")

        # Filter: drop results with similarity < 0.15 (too irrelevant)
        filtered = [
            doc for doc, dist in zip(docs, distances)
            if (1 - dist) >= 0.15
        ]

        # If all are below threshold, return top result anyway (best effort)
        return filtered if filtered else [docs[0]] if docs else []

    except Exception as e:
        print(f"[RAG] Semantic search error in '{collection.name}': {e}")
        return []

# MAIN RAG ENTRY FUNCTION
# Called by main.py -> build_prompt()

def get_rag_context(user_type: str, data: dict) -> str:
    """
    Returns a formatted context string to inject into the LLM prompt.
    Returns empty string on failure — LLM still works without it.
    """
    try:
        if user_type == "uneducated":
            query = _query_uneducated(data)
            docs  = _semantic_search(_col_vocational, query, n_results=2)

        elif user_type == "school":
            query = _query_school(data)
            docs  = _semantic_search(_col_school, query, n_results=2)

        elif user_type == "graduate":
            query = _query_graduate(data)
            docs  = _semantic_search(_col_graduate, query, n_results=2)

        elif user_type == "job_seeker":
            query = _query_job_seeker(data)
            docs  = _semantic_search(_col_graduate, query, n_results=2)

        else:
            print(f"[RAG] Unknown user_type: '{user_type}'")
            return ""

        if not docs:
            print(f"[RAG] No results for '{user_type}' — LLM will use only profile data.")
            return ""

        print(f"[RAG] Injecting {len(docs)} context chunks for '{user_type}'")
        return "\n\n---\n\n".join(docs)

    except Exception as e:
        print(f"[RAG] ERROR in get_rag_context: {e}")
        return ""