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
embedding_model_path = os.path.join("models", "all-MiniLM-L6-v2")
os.makedirs(embedding_model_path, exist_ok=True)

# Initialize FastAPI application
fastapi_app = FastAPI()

# Load embedding model with better error handling
@st.cache_resource(show_spinner=True)
def initialize_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model_identifier = "all-MiniLM-L6-v2"
        
        # Check if model exists locally
        local_model_files = ["config.json", "pytorch_model.bin", "sentence_bert_config.json"]
        has_local_model = all(os.path.exists(os.path.join(embedding_model_path, f)) for f in local_model_files)
        
        if has_local_model:
            try:
                embedder = SentenceTransformer(embedding_model_path)
                st.success("Embedder loaded from local storage")
                return embedder
            except Exception as err:
                st.warning(f"Local model loading failed: {err}")
        
        # Try to download if online
        try:
            with st.spinner("Downloading sentence transformer model (first time only)..."):
                embedder = SentenceTransformer(f'sentence-transformers/{model_identifier}')
                embedder.save(embedding_model_path)
                st.success("Model downloaded and saved locally")
                return embedder
        except Exception as download_err:
            st.error(f"Model download failed: {download_err}")
            st.info("Please download the model manually from Hugging Face and place in 'models/all-MiniLM-L6-v2'")
            st.info("Or check your internet connection and try again")
            return None
            
    except ImportError as import_err:
        st.error(f"Required package not found: {import_err}")
        return None

sentence_embedder = initialize_embedder()

# Check if embeddings are available
try:
    import faiss
    embeddings_enabled = sentence_embedder is not None
except ImportError:
    st.warning("FAISS not available - using basic matching")
    embeddings_enabled = False
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

# Setup FAISS index if available
@st.cache_data
def configure_vector_store(_records):
    if not embeddings_enabled or sentence_embedder is None:
        st.warning("Embeddings not available - using basic text matching")
        return None, [], []
    
    text_entries = [
        f"{row['Assessment Name']} {row.get('URL', '')} {row['Job Description']} {row['Scraped Description']} "
        f"{row['Job Levels']} {row['Languages']}"
        for _, row in _records.iterrows()
    ]
    
    try:
        vector_data = sentence_embedder.encode(text_entries, show_progress_bar=False)
        vector_dim = vector_data.shape[1]
        vector_index = faiss.IndexFlatL2(vector_dim)
        vector_index.add(vector_data)
        return vector_index, text_entries, vector_data
    except Exception as e:
        st.error(f"Vector store setup failed: {e}")
        return None, [], []

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
                
                # Try common description containers
                desc_selectors = [
                    'div.description', 
                    'div.job-details',
                    'section.description',
                    'div.show-more-less-html',
                    'div.description__text'
                ]
                
                content = ""
                for selector in desc_selectors:
                    sections = soup.select(selector)
                    for section in sections:
                        content += " ".join(element.text.strip() for element in section.find_all(['p', 'span', 'li']))
                
                if not content:
                    content = " ".join(soup.get_text(separator=" ").split())
                
                return content[:5000] if content else "No content found"
    except Exception as err:
        return f"Could not scrape URL: {err}"

async def find_related_assessments(user_input, limit):
    if not embeddings_enabled or faiss_index is None:
        # Basic text matching fallback
        input_lower = user_input.lower()
        scores = []
        for _, row in assessment_data.iterrows():
            text = f"{row['Assessment Name']} {row['Job Description']}".lower()
            score = sum(1 for word in input_lower.split() if word in text)
            scores.append((row, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(item, 100-score) for item, score in scores[:limit]]
    
    try:
        loop = asyncio.get_event_loop()
        input_vector = await loop.run_in_executor(
            None, 
            lambda: sentence_embedder.encode([user_input], show_progress_bar=False)
        )
        
        distances, indices = await loop.run_in_executor(
            None,
            lambda: faiss_index.search(input_vector, limit)
        )
        
        return [(assessment_data.iloc[i], float(distances[0][j])) 
                for j, i in enumerate(indices[0]) 
                if i < len(assessment_data)]
    except Exception as e:
        st.error(f"Embedding search failed: {e}")
        return []

def analyze_query_sync(input_text):
    def fetch_gemini_response(text_prompt):
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(text_prompt)
            return response
        except Exception as e:
            st.error(f"Gemini API error: {e}")
            return None

    try:
        if input_text.startswith(('http://', 'https://')):
            scraped_content = asyncio.run(scrape_url_content(input_text))
            input_text = f"URL Content: {scraped_content}\nOriginal URL: {input_text}"

        prompt = f"""
        Analyze this job description and extract key requirements:
        {input_text}
        
        Return JSON with these fields:
        - skills_needed: List of technical skills mentioned
        - time_limit: Test duration in minutes if specified, else null
        - test_categories: List of relevant test categories
        - job_tiers: List of job levels mentioned
        - preferred_langs: List of preferred languages
        
        Example output:
        {{
            "skills_needed": ["Java", "Python"],
            "time_limit": 60,
            "test_categories": ["Knowledge & Skills"],
            "job_tiers": ["Mid-Professional"],
            "preferred_langs": ["English (USA)"]
        }}
        """
        
        response = fetch_gemini_response(prompt)
        if response and response.text:
            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}
            
        # Set defaults
        result = {
            "skills_needed": parsed.get("skills_needed", []),
            "time_limit": parsed.get("time_limit"),
            "test_categories": parsed.get("test_categories", []),
            "job_tiers": parsed.get("job_tiers", []),
            "preferred_langs": parsed.get("preferred_langs", ["English (USA)"])
        }
        
        # Fallback keyword matching
        input_lower = input_text.lower()
        if not result["skills_needed"]:
            tech_keywords = ["java", "python", "sql", ".net", "c#", "javascript"]
            result["skills_needed"] = [kw for kw in tech_keywords if kw in input_lower]
            
        if not result["test_categories"]:
            if any(kw in input_lower for kw in ["knowledge", "skill", "technical"]):
                result["test_categories"] = ["Knowledge & Skills"]
            elif any(kw in input_lower for kw in ["aptitude", "ability"]):
                result["test_categories"] = ["Ability & Aptitude"]
                
        if not result["job_tiers"]:
            if "senior" in input_lower:
                result["job_tiers"] = ["Senior Professional"]
            elif "entry" in input_lower or "graduate" in input_lower:
                result["job_tiers"] = ["Graduate"]
            else:
                result["job_tiers"] = ["Mid-Professional"]
                
        # Extract time if not set
        if not result["time_limit"]:
            time_match = re.search(r"(\d+)\s*(min|minutes?|hr|hours?)", input_lower)
            if time_match:
                num = int(time_match.group(1))
                unit = time_match.group(2)
                result["time_limit"] = num * 60 if unit.startswith('h') else num
                
        return JobRequirements(**result)
        
    except Exception as err:
        st.error(f"Analysis error: {err}")
        return JobRequirements([], None, [], [], ["English (USA)"])

async def analyze_query_async(input_text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, analyze_query_sync, input_text)

async def suggest_assessments(user_input, result_count=10):
    job_specs = await analyze_query_async(user_input)
    related_entries = await find_related_assessments(user_input, limit=result_count * 2)
    suggestions = []

    for entry, distance in related_entries:
        entry_data = {
            "name": entry["Assessment Name"].lower(),
            "duration": entry["Duration"],
            "categories": [c.lower().strip() for c in entry["Test Type"].split(",")],
            "tiers": [t.lower().strip() for t in str(entry["Job Levels"]).split(",")] if entry["Job Levels"] else [],
            "langs": [l.lower().strip() for l in str(entry["Languages"]).split(",")] if entry["Languages"] else [],
            "desc": entry["Job Description"].lower(),
            "scraped": entry["Scraped Description"].lower()
        }

        # Calculate skill matches
        matched_skills = [
            skill for skill in job_specs.skills_needed 
            if (skill.lower() in entry_data["name"] or 
                skill.lower() in entry_data["desc"] or 
                skill.lower() in entry_data["scraped"])
        ]
        
        # Base score from skill matches
        score = len(matched_skills) * 100
        
        # Duration bonus
        if job_specs.time_limit and entry_data["duration"] != "N/A":
            try:
                entry_duration = float(entry_data["duration"])
                if entry_duration <= job_specs.time_limit * 1.2:
                    score += 50
            except ValueError:
                pass
                
        # Category bonus
        if job_specs.test_categories:
            matched_cats = [
                cat for cat in job_specs.test_categories 
                if any(cat.lower() in ecat for ecat in entry_data["categories"])
            ]
            score += len(matched_cats) * 30
            
        # Job level bonus
        if job_specs.job_tiers:
            matched_tiers = [
                tier for tier in job_specs.job_tiers 
                if any(tier.lower() in etier for etier in entry_data["tiers"])
            ]
            score += len(matched_tiers) * 20
            
        # Language bonus
        if job_specs.preferred_langs:
            matched_langs = [
                lang for lang in job_specs.preferred_langs 
                if any(lang.lower() in elang for elang in entry_data["langs"])
            ]
            score += len(matched_langs) * 10
            
        # Similarity bonus (if using embeddings)
        if embeddings_enabled and distance > 0:
            score += max(0, 100 - distance)
            
        if score > 0:
            suggestions.append(AssessmentSuggestion(score, entry, matched_skills))
            
    suggestions.sort(key=lambda x: x.score, reverse=True)
    return suggestions[:result_count]

# FastAPI endpoint
@fastapi_app.get("/suggest")
async def api_suggest(input_text: str, max_items: int = 10):
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text required")
    
    try:
        suggestions = await suggest_assessments(input_text, max_items)
        return {
            "suggestions": [
                {
                    "rank": idx + 1,
                    "name": sug.details["Assessment Name"],
                    "url": sug.details["URL"],
                    "score": sug.score,
                    "matched_skills": sug.matched_skills,
                    "duration": sug.details["Duration"],
                    "type": sug.details["Test Type"]
                }
                for idx, sug in enumerate(suggestions)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit UI
def run_streamlit():
    st.title("🔍 Assessment Finder")
    st.markdown("Find the best assessments for your job requirements")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        input_text = st.text_area(
            "Job Description or URL",
            height=150,
            placeholder="Paste job description or LinkedIn/Indeed URL",
            help="For best results, include required skills and job level"
        )
    with col2:
        result_count = st.slider("Results to show", 5, 20, 10)
        search_btn = st.button("Find Assessments", type="primary")
    
    if search_btn and input_text:
        with st.spinner("Analyzing job requirements..."):
            try:
                # Show requirements analysis
                requirements = asyncio.run(analyze_query_async(input_text))
                with st.expander("Extracted Requirements", expanded=True):
                    st.json(requirements.__dict__)
                
                # Show results
                st.subheader("Recommended Assessments")
                results = asyncio.run(suggest_assessments(input_text, result_count))
                
                if not results:
                    st.warning("No matching assessments found")
                    return
                
                # Display as nice cards
                for idx, result in enumerate(results, 1):
                    with st.container():
                        cols = st.columns([1, 4])
                        with cols[0]:
                            st.metric(label="Match Score", value=f"{result.score:.0f}")
                        with cols[1]:
                            st.markdown(f"**{result.details['Assessment Name']}**")
                            st.caption(f"**Type:** {result.details['Test Type']} | "
                                      f"**Duration:** {result.details['Duration']} | "
                                      f"**Levels:** {result.details['Job Levels']}")
                            
                            if result.matched_skills:
                                st.write("Matched skills: " + ", ".join(result.matched_skills))
                            
                            st.markdown(f"[View Assessment]({result.details['URL']})", unsafe_allow_html=True)
                        
                        st.divider()
                
            except Exception as e:
                st.error(f"Error generating suggestions: {e}")
    elif search_btn:
        st.warning("Please enter a job description or URL")

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI in background if needed
    fastapi_process = multiprocessing.Process(target=run_fastapi, daemon=True)
    fastapi_process.start()
    
    # Run Streamlit
    run_streamlit()