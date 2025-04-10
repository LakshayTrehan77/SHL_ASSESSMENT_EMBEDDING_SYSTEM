import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
import uvicorn
import re
import nest_asyncio
import multiprocessing

# Set Streamlit page config as the first command
st.set_page_config(page_title="Assessment Finder", layout="wide")

nest_asyncio.apply()

# Configure Gemini API
gemini_api_key = "AIzaSyCbRBKNHM-OEW7HuJ5Kogobeoop6GCzhcY"
genai.configure(api_key=gemini_api_key)

# Define model path
embedding_model_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
if not os.path.exists(embedding_model_path):
    embedding_model_path = os.path.join(os.getcwd(), "models", "all-MiniLM-L6-v2")

# Initialize FastAPI application
fastapi_app = FastAPI()

# Load embedding model
@st.cache_resource(show_spinner=True)
def initialize_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        import os
        model_identifier = "all-MiniLM-L6-v2"
        local_model_dir = os.path.join("models", model_identifier)
        os.makedirs(local_model_dir, exist_ok=True)
        expected_files = ["config.json", "pytorch_model.bin", "sentence_bert_config.json"]
        files_present = all(os.path.exists(os.path.join(local_model_dir, f)) for f in expected_files)
        
        if files_present:
            try:
                embedder = SentenceTransformer(local_model_dir)
                st.success("Embedder loaded from local storage")
                return embedder
            except Exception as err:
                st.warning(f"Local embedder failed: {err}. Trying to download...")
        
        if os.getenv('HF_HUB_OFFLINE', '0') == '0':
            with st.spinner("Fetching embedder (initial setup)..."):
                try:
                    embedder = SentenceTransformer(f'sentence-transformers/{model_identifier}')
                    embedder.save(local_model_dir)
                    st.success("Embedder fetched and stored")
                    return embedder
                except Exception as err:
                    st.error(f"Fetch failed: {err}")
                    return None
        else:
            st.warning("Offline mode: no embeddings available")
            return None
    except ImportError as err:
        st.error(f"Missing dependency: {err}")
        return None

sentence_embedder = initialize_embedder()
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    embeddings_enabled = sentence_embedder is not None
except ImportError as err:
    st.warning(f"Cannot import dependencies: {err}. Operating without embeddings.")
    embeddings_enabled = False
    SentenceTransformer = None
    faiss = None

# Load assessment data
@st.cache_data
def fetch_assessment_records():
    try:
        records = pd.read_csv("output.csv")
        records['Duration'] = records['Duration'].fillna("N/A")
        records['Job Description'] = records['Job Description'].fillna("")
        records['Job Levels'] = records['Job Levels'].fillna("")
        records['Languages'] = records['Languages'].fillna("English (USA)")
        records['Scraped Description'] = records['Scraped Description'].fillna("")
        return records
    except FileNotFoundError:
        st.error("File 'output.csv' not found. Using empty records.")
        return pd.DataFrame(columns=["Assessment Name", "URL", "Remote Testing Support", "Adaptive/IRT Support",
                                     "Test Type", "Duration", "Job Description", "Job Levels", "Languages", "Scraped Description"])
    except Exception as err:
        st.error(f"Error reading CSV: {err}")
        return pd.DataFrame(columns=["Assessment Name", "URL", "Remote Testing Support", "Adaptive/IRT Support",
                                     "Test Type", "Duration", "Job Description", "Job Levels", "Languages", "Scraped Description"])

assessment_data = fetch_assessment_records()

# Setup FAISS index
@st.cache_data
def configure_vector_store(_records):
    if not embeddings_enabled or sentence_embedder is None:
        return None, [], []
    text_entries = [
        f"{row['Assessment Name']} {row.get('URL', '')} {row['Job Description']} {row['Scraped Description']} "
        f"{row['Job Levels']} {row['Languages']} {' '.join(row['Assessment Name'].lower().split())}"
        for _, row in _records.iterrows()
    ]
    vector_data = sentence_embedder.encode(text_entries, show_progress_bar=False)
    vector_dim = vector_data.shape[1]
    vector_index = faiss.IndexFlatL2(vector_dim) if faiss else None
    if vector_index:
        vector_index.add(vector_data)
    return vector_index, text_entries, vector_data

faiss_index, assessment_texts, assessment_embeddings = configure_vector_store(assessment_data)

# Define dataclasses
@dataclass
class JobRequirements:
    skills_needed: list
    time_limit: float
    test_categories: list
    job_tiers: list
    preferred_langs: list

@dataclass
class AssessmentSuggestion:
    score: float
    details: pd.Series
    matched_skills: list

# Asynchronous utility functions
async def scrape_url_content(url):
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10) as response:
                response.raise_for_status()
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                desc_sections = soup.find_all('div', class_=['description', 'job-details', 'show-more-less-html', 'description__text']) or \
                                soup.find_all('section', class_=['description'])
                content = ""
                for section in desc_sections:
                    content += " ".join(p.text.strip() for p in section.find_all('p') if p.text.strip())
                    content += " ".join(span.text.strip() for span in section.find_all('span') if span.text.strip())
                    content += " ".join(li.text.strip() for li in section.find_all('li') if li.text.strip())
                    content += " ".join(script.text.strip() for script in section.find_all('script') if 'description' in script.text.lower())
                if not content:
                    content = " ".join(soup.body.get_text(separator=" ").strip().split())
                return content if content else "No content found"
    except Exception as err:
        return f"URL scrape error {url}: {err}"

async def find_related_assessments(user_input, limit):
    if not embeddings_enabled or faiss_index is None or sentence_embedder is None:
        return [(row, 0) for _, row in assessment_data.iterrows()]
    
    loop = asyncio.get_event_loop()
    # Encode the input text
    input_vector = await loop.run_in_executor(
        None, 
        lambda: sentence_embedder.encode([user_input], show_progress_bar=False)
    )
    
    # Search the FAISS index
    distances, indices = await loop.run_in_executor(
        None,
        lambda: faiss_index.search(input_vector, limit)
    )
    
    return [(assessment_data.iloc[i], float(distances[0][j])) 
            for j, i in enumerate(indices[0]) 
            if i < len(assessment_data)]

def analyze_query_sync(input_text):
    def fetch_gemini_response(text_prompt):
        config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }
        model_instance = genai.GenerativeModel(model_name="models/gemini-2.0-flash", generation_config=config)
        return model_instance.generate_content(text_prompt)

    try:
        if input_text.startswith(('http://', 'https://')):
            input_text = asyncio.run(scrape_url_content(input_text))
        related_items = asyncio.run(find_related_assessments(input_text, 5))
        context_info = "\n".join([
            f"Assessment: {item['Assessment Name']}, Type: {item['Test Type']}, Duration: {item['Duration']}, "
            f"Job Description: {item['Job Description']}, Job Levels: {item['Job Levels']}, Languages: {item['Languages']}"
            for item, _ in related_items
        ])
        analysis_prompt = f"""
        Analyze as a job assessment expert. Extract from the query and context:
        - Skills needed (e.g., Java, Python, SQL, .NET Framework, explicit or inferred).
        - Time limit in minutes (e.g., '1 hour' → 60, null if unclear).
        - Test categories: ['Ability & Aptitude', 'Assessment Exercises', 'Biodata & Situational Judgement',
          'Competencies', 'Development & 360', 'Knowledge & Skills', 'Personality & Behavior', 'Simulations'].
        - Job tiers (e.g., 'Graduate', 'Mid-Professional', inferred or explicit).
        - Preferred languages (e.g., 'English (USA)', inferred or explicit).
        Return JSON with 'skills_needed', 'time_limit', 'test_categories', 'job_tiers', 'preferred_langs'.
        Defaults: skills_needed: [], time_limit: null, test_categories: [], job_tiers: [], preferred_langs: ['English (USA)'].
        Query: {input_text}
        Context: {context_info}
        """
        response = fetch_gemini_response(analysis_prompt)
        parsed_data = json.loads(response.text) if response.text and response.text.strip().startswith("{") else {
            "skills_needed": [], "time_limit": None, "test_categories": [], "job_tiers": [], "preferred_langs": ["English (USA)"]
        }
        if not parsed_data.get("skills_needed"):
            parsed_data["skills_needed"] = [s for s in ["java", "python", "sql", ".net framework"] if s in input_text.lower()]
        if not parsed_data.get("test_categories"):
            parsed_data["test_categories"] = ["Knowledge & Skills"] if any(s in input_text.lower() for s in ["java", "python", "sql", ".net"]) else []
        if not parsed_data.get("time_limit") and "minutes" in input_text.lower():
            match = re.search(r"(\d+)\s*minutes?", input_text.lower())
            parsed_data["time_limit"] = int(match.group(1)) if match else None
        if not parsed_data.get("job_tiers"):
            parsed_data["job_tiers"] = ["Mid-Professional"] if "experienced" in input_text.lower() else []
        if not parsed_data.get("preferred_langs"):
            parsed_data["preferred_langs"] = ["English (USA)"]
        return JobRequirements(
            skills_needed=parsed_data["skills_needed"],
            time_limit=parsed_data["time_limit"],
            test_categories=parsed_data["test_categories"],
            job_tiers=parsed_data["job_tiers"],
            preferred_langs=parsed_data["preferred_langs"]
        )
    except Exception as err:
        st.error(f"Query analysis failed: {err}")
        return JobRequirements([], None, [], [], ["English (USA)"])

async def analyze_query_async(input_text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, analyze_query_sync, input_text)

async def suggest_assessments(user_input, result_count=10):
    job_specs = await analyze_query_async(user_input)
    related_entries = await find_related_assessments(user_input, limit=result_count * 2)
    suggestions_list = []

    for entry, distance in related_entries:
        entry_name = entry["Assessment Name"].lower()
        time_duration = entry["Duration"]
        categories = [c.lower().strip() for c in entry["Test Type"].split(", ")]
        tiers = [t.lower().strip() for t in str(entry["Job Levels"]).split(", ")] if entry["Job Levels"] else []
        langs = [l.lower().strip() for l in str(entry["Languages"]).split(", ")] if entry["Languages"] else []
        desc_text = entry["Job Description"].lower()
        scraped_text = entry["Scraped Description"].lower()

        skill_matches = [skill for skill in job_specs.skills_needed if skill in entry_name or skill in desc_text or skill in scraped_text or any(skill in c for c in categories)]
        match_count = len(skill_matches)
        primary_score = 1000 if match_count == len(job_specs.skills_needed) and job_specs.skills_needed else match_count * 200

        bonus_score = 0
        if job_specs.time_limit and time_duration != "N/A" and float(time_duration) <= float(job_specs.time_limit) * 1.2:
            bonus_score += 50
        elif not job_specs.time_limit and time_duration != "N/A" and float(time_duration) <= 60:
            bonus_score += 25

        if job_specs.test_categories and any(cat in categories for cat in job_specs.test_categories):
            bonus_score += 40
        elif not job_specs.test_categories and any(c in ["knowledge & skills", "ability & aptitude"] for c in categories):
            bonus_score += 20

        if job_specs.job_tiers and any(tier in tiers for tier in job_specs.job_tiers):
            bonus_score += 30
        elif not job_specs.job_tiers and "mid-professional" in tiers:
            bonus_score += 10

        if job_specs.preferred_langs and any(lang in langs for lang in job_specs.preferred_langs):
            bonus_score += 20
        elif not job_specs.preferred_langs and "english (usa)" in langs:
            bonus_score += 5

        similarity_value = 0 if not embeddings_enabled else (100 - (distance / np.max(distance) * 50) if distance > 0 else 0)
        bonus_score += similarity_value * 0.5

        total_value = primary_score + bonus_score
        if total_value > 0:
            suggestions_list.append(AssessmentSuggestion(total_value, entry, skill_matches))

    suggestions_list.sort(key=lambda x: x.score, reverse=True)
    return suggestions_list[:result_count]

# FastAPI endpoint
@fastapi_app.get("/suggest")
async def fetch_suggestions(input_text: str, max_items: int = 10):
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is required")
    suggestion_results = await suggest_assessments(input_text, max_items)
    return {
        "suggestions": [
            {
                "position": i + 1,
                "assessment_title": sug.details["Assessment Name"],
                "link": sug.details["URL"],
                "duration": float(sug.details["Duration"]) if sug.details["Duration"] != "N/A" else "N/A",
                "remote_support": sug.details["Remote Testing Support"],
                "adaptive_support": sug.details["Adaptive/IRT Support"],
                "category": sug.details["Test Type"],
                "tiers": sug.details["Job Levels"],
                "langs": sug.details["Languages"],
                "description": sug.details["Job Description"],
                "value": sug.score
            }
            for i, sug in enumerate(suggestion_results)
        ]
    }

# Streamlit app with new UI
def run_streamlit():
    st.title("Assessment Finder")
    st.write("Enter a job description or URL to get assessment recommendations.")

    # Input selection
    entry_mode = st.selectbox("Input Type", ["Text", "URL"], index=1)
    item_limit = st.slider("Number of Suggestions", 5, 15, 10)
    user_entry = st.text_area(
        "Job Details or URL",
        height=100,
        placeholder="E.g., 'Need Java, Python, SQL devs with .NET, 40 min' or a URL",
        value="https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in" if entry_mode == "URL" else "Hiring Java, Python, and SQL developers with .NET Framework experience, 40 minutes."
    )

    if st.button("Get Suggestions"):
        if user_entry:
            with st.spinner("Generating suggestions..."):
                # Show parsed requirements
                st.subheader("Parsed Requirements")
                needs = asyncio.run(analyze_query_async(user_entry))
                st.json(needs.__dict__)

                # Show recommendations
                st.subheader("Recommendations")
                matches = asyncio.run(suggest_assessments(user_entry, item_limit))
                if matches:
                    display_data = [
                        {
                            "Rank": i + 1,
                            "Name": m.details["Assessment Name"],
                            "URL": f'<a href="{m.details["URL"]}" target="_blank">Link</a>',
                            "Duration": float(m.details["Duration"]) if m.details["Duration"] != "N/A" else "N/A",
                            "Test Type": m.details["Test Type"],
                            "Description": m.details["Job Description"][:100] + "..." if len(m.details["Job Description"]) > 100 else m.details["Job Description"]
                        }
                        for i, m in enumerate(matches)
                    ]
                    matches_df = pd.DataFrame(display_data)
                    st.write(matches_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.error("No assessments found. Check your input.")
        else:
            st.warning("Please provide an input.")

async def assess_performance():
    test_cases = [
        {"input": "Hiring Java, Python, SQL developers with .NET Framework, 40 minutes, Mid-Professional, English (USA)",
         "expected": [".NET Framework 4.5", "Java Programming", "Python Programming", "SQL Server"]},
        {"input": "Research Engineer AI, 60 minutes, Professional Individual Contributor, English (USA)",
         "expected": ["AI Skills", "Aeronautical Engineering"]}
    ]
    top_n = 5
    recall_values = []
    precision_values = []

    for case in test_cases:
        results = await suggest_assessments(case["input"], top_n)
        result_titles = [r.details["Assessment Name"] for r in results]
        expected_set = set(case["expected"])
        matches_found = len(set(result_titles) & expected_set)
        total_expected = len(expected_set)
        recall = matches_found / total_expected if total_expected > 0 else 0
        recall_values.append(recall)
        precision_sum = 0
        relevant_hits = 0
        for pos, title in enumerate(result_titles[:top_n], 1):
            if title in expected_set:
                relevant_hits += 1
                precision_sum += relevant_hits / pos
        avg_precision = precision_sum / min(top_n, total_expected) if min(top_n, total_expected) > 0 else 0
        precision_values.append(avg_precision)

    return np.mean(recall_values) if recall_values else 0, np.mean(precision_values) if precision_values else 0

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI in a separate process
    fastapi_process = multiprocessing.Process(target=run_fastapi)
    fastapi_process.start()
    
    # Run Streamlit in the main process
    run_streamlit()