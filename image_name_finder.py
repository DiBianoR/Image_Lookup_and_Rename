import os
import sys
import requests
import json
import urllib.parse
import re
import time
from pathlib import Path
from google import genai
from google.cloud import vision
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# ==========================================
# Configuration
# ==========================================
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DIRECTORY_PATH = os.getenv("DIRECTORY_PATH")
BASE_URL = os.getenv("BASE_URL")
IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "serpapi").lower()
ENABLE_PASS_2 = os.getenv("ENABLE_PASS_2", "True").lower() in ['true', '1', 't', 'y', 'yes']

# Safety check for distributed use
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
    print("[!] ERROR: Missing Google API Key.")
    print("    Please open the .env file and paste your Google API key.")
    sys.exit(1)

if IMAGE_BACKEND not in ["serpapi", "vision"]:
    print(f"[!] ERROR: Invalid IMAGE_BACKEND '{IMAGE_BACKEND}'.")
    print("    Please open the .env file and set IMAGE_BACKEND to either 'serpapi' or 'vision'.")
    sys.exit(1)

if IMAGE_BACKEND == "serpapi" and (not SERPAPI_KEY or SERPAPI_KEY == "your_serpapi_key_here"):
    print("[!] ERROR: Missing SerpApi Key.")
    print("    Please open the .env file and paste your SerpApi key, or switch IMAGE_BACKEND to 'vision'.")
    sys.exit(1)

if IMAGE_BACKEND == "vision" and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print("[!] ERROR: Missing Google Cloud Vision Credentials.")
    print("    Please open the .env file and configure GOOGLE_APPLICATION_CREDENTIALS.")
    sys.exit(1)

if not DIRECTORY_PATH or not BASE_URL or DIRECTORY_PATH == "your_image_directory_here" or BASE_URL == "your_image_directory_url_here":
    print("[!] ERROR: Missing Directory or Base URL.")
    print("    Please open the .env file and configure your DIRECTORY_PATH and BASE_URL.")
    sys.exit(1)

# Ensure the Base URL always ends with a slash so urllib doesn't mangle the file paths
if not BASE_URL.endswith('/'):
    BASE_URL += '/'

# --- PROCESSING SWITCHES ---
RENAME_FILES = True  # Set to False to keep original filenames
TITLE_UNKNOWNS_BY_SUBJECT = True  # just give it a name based on what we see if we can't find the name
SCRUB_EMOJIS_FROM_LLM_INPUT = True  # Set to True to strip emojis before sending data to Gemini (prevents some crashes)
SAVE_RAW_LENS_DATA = True  # Set to True to save the raw SerpApi response to a '_tmp.json' file
REDO_LLM = False  # Set to True to re-run Gemini on files that already have a final .json metadata file
PASSWORD=GORILLE
# Globally disable safety filters for art processing
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

LENS_QUERY_TEMP = 1.0
SIMPLE_SEARCH_TEMP = 0.85
STRING_EXTRACTION_TEMP = 0.1

# Regex to match Python's escaped Unicode surrogate pairs (e.g., \uD83D\uDE00)
# and standard escaped Unicode characters in the upper ranges (e.g., \u2694)
ESCAPED_EMOJI_PATTERN = re.compile(
    r'(\\u[dD][a-fA-F0-9]{3}\\u[dD][a-fA-F0-9]{3}|\\u2[0-9a-fA-F]{3}|\\u[fF][0-9a-fA-F]{3})')

# Initialize the Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define the constrained JSON schema for the primary artwork extraction
GEMINI_SCHEMA = {
    "type": "object",
    "properties": {
        "query_response": {
            "type": "string",
            "description": "Full response to the query explaining the reasoning and how the title was determined."
        },
        "title_found": {
            "type": "boolean",
            "description": "Was the title definitively found? If there were multiple likely options, this is False."
        },
        "title_of_work": {
            "type": ["string", "null"],
            "description": "Title of Work if it was found, null otherwise."
        },
        "artist_name": {
            "type": ["string", "null"],
            "description": "Artist Name if it was found, null otherwise."
        },
        "artist_first_name": {
            "type": ["string", "null"],
            "description": "The artist's first name if known, null otherwise, and null if they use a mononym."
        },
        "artist_last_name": {
            "type": ["string", "null"],
            "description": "The artist's last name if known, null otherwise. If the artist goes by a single name or pseudonym (e.g., 'Brom'), place it here."
        },
        "year_of_completion": {
            "type": ["integer", "null"],
            "description": "Year of Completion if it was found, null otherwise. Extract as a 4-digit integer."
        },
        "source": {
            "type": ["string", "null"],
            "description": "Source Publication, Publication Context, or Original Appearance - where the piece made its public debut. Or null if not found."
        },
        "format": {
            "type": ["string", "null"],
            "description": "This describes the specific function the artwork was commissioned to serve within the publication - Cover Art, Magazine Cover, Book Cover, Module Wrap, Interior Illustration, Promotional, Packaging, Private Commission, etc. Or null if not found."
        },
        "medium": {
            "type": ["string", "null"],
            "description": "Medium (e.g., Oil on canvas) if it was found, null otherwise."
        },
        "dimensions": {
            "type": ["string", "null"],
            "description": "Physical Dimensions of the physical painting (usually height x width, or height x width x depth) if found, null if digital art or not found."
        },
        "title_guess": {
            "type": ["string", "null"],
            "description": "Title of work if definitively found, a best guess if a likely title was found but it was not definitive, If there were multiple likely options, prioritize ai_overview's guess, 'source' or 'source - format' sometimes can work as a title if you have source[above], null otherwise."
        },
        "subject": {
            "type": ["string", "null"],
            "description": "Title aside, [if it was mentioned in the search results] what is this an image of? Null if it is unclear or not mentioned."
        }
    },
    "required": [
        "query_response", "title_found", "title_of_work", "artist_name", "year_of_completion",
        "source", "format", "medium", "dimensions", "title_guess", "subject"
    ]
}

# Define the separate schemas for the targeted fallbacks
ARTIST_FOLLOWUP_SCHEMA = {
    "type": "object",
    "properties": {
        "artist_name": {
            "type": ["string", "null"],
            "description": "Artist Name if explicitly stated in the text, null otherwise."
        },
        "artist_first_name": {
            "type": ["string", "null"],
            "description": "The artist's first name if known, null otherwise, and null if they use a mononym."
        },
        "artist_last_name": {
            "type": ["string", "null"],
            "description": "The artist's last name if known, null otherwise. If the artist goes by a single name or pseudonym (e.g., 'Brom'), place it here."
        }
    },
    "required": ["artist_name", "artist_first_name", "artist_last_name"]
}

YEAR_FOLLOWUP_SCHEMA = {
    "type": "object",
    "properties": {
        "year_of_completion": {
            "type": ["integer", "null"],
            "description": "Year of Completion if explicitly stated in the text, null otherwise. Extract as a 4-digit integer."
        }
    },
    "required": ["year_of_completion"]
}


def sanitize_filename(name: str) -> str:
    """Removes characters that are illegal in Windows file paths, including hidden control/null bytes."""
    # \x00-\x1f catches all invisible control characters (including the null byte \x00)
    safe_name = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", name)
    return safe_name.strip()


def clean_redundant_title_text(title: str, first: str, last: str, year: int) -> str:
    """Strips redundant artist names from the start and years from the end of a title string."""
    if not title:
        return title

    cleaned = title.strip()

    # 1. Strip Year from the end
    if year:
        y_str = str(year)
        # Matches: " 1984", " - 1984", " (1984)", " [1984]", ", 1984" exactly at the end of the string
        year_pattern = r'\s*[\,\-]?\s*[\(\[]?' + y_str + r'[\)\]]?\s*$'
        cleaned = re.sub(year_pattern, '', cleaned, flags=re.IGNORECASE)

    # 2. Strip Artist from the beginning
    artist_patterns = []
    if first and last:
        artist_patterns.append(re.escape(f"{first} {last}"))
        artist_patterns.append(re.escape(f"{last}, {first}"))
        artist_patterns.append(re.escape(f"{last} {first}"))
    elif last:
        artist_patterns.append(re.escape(last))
    elif first:
        artist_patterns.append(re.escape(first))

    if artist_patterns:
        # Combine patterns: Matches any of the artist formats at the start of the string, plus trailing separators
        combined_artists = "|".join(artist_patterns)
        artist_regex = r'^(' + combined_artists + r')\s*[\,\-]?\s*'
        cleaned = re.sub(artist_regex, '', cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def get_unique_base_name(directory: Path, desired_base: str, original_filepath: Path) -> str:
    """
    Mimics Windows file duplication handling using square brackets [1], [2].
    """
    counter = 1
    current_base = desired_base

    while True:
        test_img = directory / f"{current_base}{original_filepath.suffix}"
        test_raw = directory / f"{current_base}_tmp.json"
        test_meta = directory / f"{current_base}.json"

        is_safe = (
                (not test_img.exists() or test_img == original_filepath) and
                (not test_raw.exists() or test_raw == original_filepath.with_name(
                    f"{original_filepath.stem}_tmp.json")) and
                (not test_meta.exists() or test_meta == original_filepath.with_suffix('.json'))
        )

        if is_safe:
            return current_base

        current_base = f"{desired_base} [{counter}]"
        counter += 1


def scrub_lens_data(data: dict) -> bool:
    """
    Specifically targets and removes tracking data and useless token-bloat
    (like pricing, reviews, and image URLs) strictly from the visual_matches list.
    """
    modified = False

    # 1. Remove top-level tracking blocks
    for top_key in ["search_metadata", "search_parameters", "lens_detect_zones"]:
        if top_key in data:
            del data[top_key]
            modified = True

    # The exact keys we want to purge from individual match items
    match_keys_to_purge = {
        "source_icon", "price", "in_stock", "rating", "reviews"
        # ,"thumbnail", "thumbnail_width", "thumbnail_height", "image", "image_width", "image_height"
    }

    # 2. Target specific fields exactly one level deep inside matches
    for list_key in ["visual_matches", "shopping_results"]:
        if list_key in data and isinstance(data[list_key], list):
            for match in data[list_key]:
                # We use list(match.keys()) so we can delete keys while iterating
                for key in list(match.keys()):
                    if key in match_keys_to_purge:
                        del match[key]
                        modified = True

    return modified


def search_google_lens(image_url: str) -> dict:
    """Calls SerpApi Google Lens endpoint with explicit configuration."""
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
        "type": "all",
        "auto_crop": "false"
    }

    # Optional text query logic (skipped for now while empty)
    text_query = ""
    if text_query:
        params["q"] = text_query

    response = requests.get("https://serpapi.com/search", params=params)
    response.raise_for_status()
    return response.json()


def search_google_vision(image_url: str) -> dict:
    """Calls Google Cloud Vision API Web Detection and formats the output cleanly."""
    vision_client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = image_url

    response = vision_client.web_detection(image=image)
    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")

    annotations = response.web_detection

    # Format into a clean, LLM-friendly dictionary
    vision_data = {
        "web_entities": [{"description": entity.description, "score": entity.score}
                         for entity in annotations.web_entities if entity.description],
        "pages_with_matching_images": [{"url": page.url, "page_title": page.page_title}
                                       for page in annotations.pages_with_matching_images if page.page_title],
        "best_guess_labels": [label.label for label in annotations.best_guess_labels]
    }
    return vision_data


def generate_vision_overview(vision_data: dict, model_name: str = "gemini-3.1-flash-lite-preview") -> str:
    """Uses Gemini to synthesize an ai_overview paragraph from raw Vision data."""
    if not vision_data.get("web_entities") and not vision_data.get("pages_with_matching_images"):
        return ""

    prompt = f"""
Given the following raw web detection data for an image, write a single concise paragraph summarizing what this image is. 
Focus on identifying the artist, title, subject matter, and year if available in the data.
Do not mention the data structure, just summarize the facts as if you were an encyclopedia.

Raw Data:
{json.dumps(vision_data, indent=2)}
"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"temperature": 0.2}
    )
    return response.text


def fetch_full_ai_overview(ai_overview_dict: dict, error_list: list, filename: str) -> dict:
    """
    Takes the lazily-loaded ai_overview dict, uses the page_token to fetch the full text.
    Retries up to 3 times if Google returns an empty object.
    """
    page_token = ai_overview_dict.get("page_token")
    if not page_token:
        return ai_overview_dict

    print("[*] Found AI Overview token. Fetching full text (may take a few seconds)...")
    params = {
        "engine": "google_ai_overview",
        "page_token": page_token,
        "api_key": SERPAPI_KEY
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            new_data = response.json()
            fetched_overview = new_data.get("ai_overview", {})

            if fetched_overview:
                print("    [+] Successfully retrieved full AI Overview.")
                return fetched_overview
            else:
                print(f"    [-] Attempt {attempt + 1}: AI Overview returned empty. Retrying...")
                time.sleep(3)

        except Exception as e:
            print(f"    [-] Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    err_msg = f"Failed to fetch AI Overview for {filename} after {max_retries} attempts."
    print(f"    [!] {err_msg}")
    error_list.append(err_msg)
    return {}  # Wipe the token out since it failed


def analyze_json_with_gemini(lens_json_data: dict, model_name: str = "gemini-2.5-flash-lite", temp: float = 1.0) -> str:
    """Passes the raw Lens JSON to Gemini for structured extraction. Defaults to the cheapest model"""

    raw_json_str = json.dumps(lens_json_data)  # Python will escape emojis into ASCII strings

    if SCRUB_EMOJIS_FROM_LLM_INPUT:
        sanitized_json_str = ESCAPED_EMOJI_PATTERN.sub('', raw_json_str)
    else:
        sanitized_json_str = raw_json_str

    prompt = f"```json\n{sanitized_json_str}\n```\n"
    prompt += """\
Using only this json, without looking online or at any other sources, can you definitively say what the title of the image in question is?
Titles from arthive are not reliable."""

    # 1. Define your base configuration that works for all models
    gen_config = {
        "response_mime_type": "application/json",
        "response_json_schema": GEMINI_SCHEMA,
        "temperature": temp,
        "safety_settings": DEFAULT_SAFETY_SETTINGS
    }

    # 2. Check the model name and bump the thinking level up one notch ("low")
    # This prevents the script from crashing if you swap to a model like 2.5-flash-lite
    # that doesn't support this specific thinking parameter, or 3-flash which defaults to "high".
    if model_name == "gemini-3.1-flash-lite-preview":
        gen_config["thinking_config"] = {"thinking_level": "low"}  # minimal -> low -> medium -> high

    # 3. Pass the dynamic config dict to the client
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=gen_config,
    )
    return response.text


def search_missing_artist(title: str, current_metadata: dict, model_name: str = "gemini-2.5-flash-lite") -> dict:
    """Two-step process: Searches Google for the artist, then extracts to JSON."""

    search_prompt = f"""\
We have identified an artwork titled '{title}'. Here is the current known metadata:
{json.dumps(current_metadata)}

Using Google Search, find the name of the original artist for this specific artwork. Provide a brief text summary of your findings."""

    search_response = client.models.generate_content(
        model=model_name,
        contents=search_prompt,
        config={"temperature": SIMPLE_SEARCH_TEMP, "tools": [{"google_search": {}}]},
    )

    extract_prompt = f"""\
We have identified an artwork titled '{title}'. Using Google Search, find the name of the original artist for this specific artwork. Provide a brief text summary of your findings.
[search completed]
summary_of_findings:
{search_response.text}
Based on the summary, extract the artist's name. If there is confusion, or no specific artist is confirmed, return null."""

    extract_response = client.models.generate_content(
        model=model_name,
        contents=extract_prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": ARTIST_FOLLOWUP_SCHEMA,
            "temperature": STRING_EXTRACTION_TEMP
        },
    )
    return json.loads(extract_response.text)


def search_missing_year(title: str, current_metadata: dict, model_name: str = "gemini-2.5-flash-lite") -> dict:
    """Two-step process: Searches Google for the year, then extracts to JSON."""

    search_prompt = f"""\
We have identified an artwork titled '{title}'. Here is the current known metadata:
{json.dumps(current_metadata)}

Using Google Search, find the year of completion or original publication for this specific artwork. Provide a brief text summary of your findings."""

    search_response = client.models.generate_content(
        model=model_name,
        contents=search_prompt,
        config={"temperature": SIMPLE_SEARCH_TEMP, "tools": [{"google_search": {}}]},
    )

    extract_prompt = f"""
We have identified an artwork titled '{title}'. Using Google Search, find the year of completion or original publication for this specific artwork. Provide a brief text summary of your findings.
[search completed]
summary_of_findings:
{search_response.text}
Based on the summary, extract the year of completion. If there is confusion, or no specific year is confirmed, return null."""

    extract_response = client.models.generate_content(
        model=model_name,
        contents=extract_prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": YEAR_FOLLOWUP_SCHEMA,
            "temperature": STRING_EXTRACTION_TEMP
        },
    )
    return json.loads(extract_response.text)


def main():
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_dir = Path(DIRECTORY_PATH)

    # Track non-fatal errors to report at the end
    run_errors = []
    skipped_files = []

    if not image_dir.exists():
        print(f"[!] Cannot find the directory: {DIRECTORY_PATH}")
        return

    files_to_process = [f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    # Calculate exactly how many files will trigger processing
    active_files = 0
    for filepath in files_to_process:
        if filepath.with_suffix('.json').exists() and not REDO_LLM:
            continue
        active_files += 1

    if active_files == 0:
        print("[*] No new files to process.")
        return

    # --- COST ESTIMATION LOGIC ---
    COST_IMAGE_SEARCH_SERPAPI = 0.025
    COST_IMAGE_SEARCH_VISION = 0.0055  # 0.0035 Vision + 0.0020 LLM overview
    COST_PASS1 = 0.00775
    COST_PASS2 = 0.02300
    COST_FALLBACK_TOKENS = 0.0019 * 2  # Gemini 2.5 Flash-Lite
    GROUNDING_SEARCH_COST = 0.035  # $35 per 1000 after 1500 free queries

    total_image_search_cost = active_files * (
        COST_IMAGE_SEARCH_SERPAPI if IMAGE_BACKEND == "serpapi" else COST_IMAGE_SEARCH_VISION)
    total_pass1_cost = active_files * COST_PASS1
    total_pass2_cost = active_files * COST_PASS2 if ENABLE_PASS_2 else 0.0

    # Upper limit assumes ALL active files need BOTH fallbacks (2 searches per file)
    total_fallback_searches = active_files * 2
    free_searches_used = min(1500, total_fallback_searches)
    paid_searches = total_fallback_searches - free_searches_used

    total_fallback_token_cost = active_files * COST_FALLBACK_TOKENS
    total_fallback_search_cost = paid_searches * GROUNDING_SEARCH_COST
    total_fallback_cost = total_fallback_token_cost + total_fallback_search_cost

    total_max_cost = total_image_search_cost + total_pass1_cost + total_pass2_cost + total_fallback_cost

    print("\n" + "=" * 50)
    print("COST ESTIMATION (UPPER LIMIT)")
    print("=" * 50)
    print(f"Backend Engine:          {IMAGE_BACKEND.upper()}")
    print(f"Active files to process: {active_files}")
    print(f"  - Max Image Search:    ${total_image_search_cost:.4f}")
    print(f"  - Max Pass 1 Cost:     ${total_pass1_cost:.4f}")
    if ENABLE_PASS_2:
        print(f"  - Max Pass 2 Cost:     ${total_pass2_cost:.4f}")
    print(
        f"  - Max Fallback Cost:   ${total_fallback_cost:.4f} ({free_searches_used} free API searches, {paid_searches} paid)")
    print("-" * 50)
    print(f"Max Total Batch Cost:    ${total_max_cost:.4f}")
    print("=" * 50)

    user_input = input("Proceed with processing? (y/n): ")
    if user_input.lower() not in ['y', 'yes']:
        print("[*] Processing aborted by user.")
        sys.exit()

    # --- MAIN PROCESSING LOOP ---
    for filepath in files_to_process:
        print(f"\n" + "=" * 50)
        print(f"Processing: {filepath.name}")
        print("=" * 50)

        # Explicitly separate the raw SerpApi JSON from the parsed Metadata JSON
        raw_lens_filepath = filepath.with_name(f"{filepath.stem}_tmp.json")
        metadata_filepath = filepath.with_suffix('.json')

        # Check if we should skip this file based on REDO_LLM flag
        if metadata_filepath.exists() and not REDO_LLM:
            print(f"[*] Found existing metadata: {metadata_filepath.name}")
            print("    Skipping. (Set REDO_LLM=True to reprocess local data).")
            skipped_files.append(filepath.name)
            continue

        try:
            # 1. Handle Image Search Data
            needs_api_fetch = True

            if raw_lens_filepath.exists():
                with open(raw_lens_filepath, "r", encoding="utf-8") as f:
                    lens_data = json.load(f)

                if IMAGE_BACKEND == "serpapi":
                    # Scrub existing cache to prevent filename data leakage
                    cache_modified = scrub_lens_data(lens_data)

                    # If a local cache contains a page_token, OR if the ai_overview is literally {}, it is expired/failed.
                    if "ai_overview" in lens_data and (
                            not lens_data["ai_overview"] or "page_token" in lens_data["ai_overview"]):
                        warn_msg = f"Local cache contained an expired or empty AI Overview for {filepath.name}. Discarding cache."
                        print(f"    [!] {warn_msg}")
                        run_errors.append(warn_msg)
                        raw_lens_filepath.unlink()  # Delete the expired/failed cache file
                    else:
                        print(f"[*] Found existing raw data: {raw_lens_filepath.name} (Skipping Search)")
                        needs_api_fetch = False

                        # If we modified the cache by scrubbing out the URLs, save it so it's clean for future runs
                        if cache_modified and SAVE_RAW_LENS_DATA:
                            with open(raw_lens_filepath, "w", encoding="utf-8") as f:
                                json.dump(lens_data, f, indent=4)
                            print("    [*] Scrubbed original filename leak from existing local cache.")
                else:
                    # Vision backend cache logic
                    print(f"[*] Found existing Vision data: {raw_lens_filepath.name} (Skipping Search)")
                    needs_api_fetch = False

            if needs_api_fetch:
                safe_filename = urllib.parse.quote(filepath.name)
                public_url = f"{BASE_URL}{safe_filename}"
                print(f"[*] Generated URL: {public_url}")

                if IMAGE_BACKEND == "serpapi":
                    print("[*] Querying SerpApi Google Lens...")
                    lens_data = search_google_lens(public_url)

                    # Scrub the fresh data before anything else sees it
                    scrub_lens_data(lens_data)

                    # Intercept and fetch the full AI Overview immediately before it expires
                    if "ai_overview" in lens_data:
                        lens_data["ai_overview"] = fetch_full_ai_overview(lens_data["ai_overview"], run_errors,
                                                                          filepath.name)

                elif IMAGE_BACKEND == "vision":
                    print("[*] Querying Google Cloud Vision...")
                    lens_data = search_google_vision(public_url)

                    print("[*] Synthesizing AI Overview from Vision Data...")
                    synthesized_overview = generate_vision_overview(lens_data)
                    lens_data["ai_overview"] = {"text_blocks": [{"snippet": synthesized_overview}]}

                if SAVE_RAW_LENS_DATA:
                    with open(raw_lens_filepath, "w", encoding="utf-8") as json_file:
                        json.dump(lens_data, json_file, indent=4)
                    print(f"[*] Saved raw data to: {raw_lens_filepath.name}")

            # Assess if the AI Overview was completely empty or failed
            missing_ai_overview = not bool(lens_data.get("ai_overview"))

            # 2. Pass the data to Gemini for extraction
            print("[*] Passing data to Gemini for analysis (Pass 1: flash-lite)...")
            gemini_output = analyze_json_with_gemini(lens_data, model_name="gemini-3.1-flash-lite-preview",
                                                     temp=LENS_QUERY_TEMP)
            parsed_metadata = json.loads(gemini_output)

            # Evaluate First Pass
            title_val = parsed_metadata.get("title_of_work")
            title_found = parsed_metadata.get("title_found") and title_val
            is_definitive = title_found

            # 2b. Second Pass if Title not found AND Pass 2 is enabled
            if not title_found and ENABLE_PASS_2:
                print("[*] Title not found definitively. Retrying (Pass 2: flash)...")
                gemini_output_pass2 = analyze_json_with_gemini(lens_data, model_name="gemini-3-flash-preview",
                                                               temp=LENS_QUERY_TEMP)
                parsed_metadata = json.loads(gemini_output_pass2)

                # Re-evaluate
                title_val = parsed_metadata.get("title_of_work")
                title_found = parsed_metadata.get("title_found") and title_val
                is_definitive = title_found
            elif not title_found and not ENABLE_PASS_2:
                print("    [*] Skipping Pass 2 (Disabled in config).")

            # 2c. Fallback to title_guess or subject(Working Title)
            if not title_found:
                if parsed_metadata.get("title_guess"):
                    title_val = f"{parsed_metadata.get('title_guess')} (WT)"
                    title_found = True
                    is_definitive = False
                    print(f"    [+] Falling back to Working Title: {title_val}")
                elif TITLE_UNKNOWNS_BY_SUBJECT and parsed_metadata.get("subject"):
                    title_val = f"{parsed_metadata.get('subject')} (WT)"
                    title_found = True
                    is_definitive = False
                    print(f"    [+] Falling back to Image Subject: {title_val}")

            # 2d. OVERRIDE: If we didn't get an AI Overview, force (WT) classification on whatever title we found
            if title_found and missing_ai_overview:
                is_definitive = False
                if not title_val.endswith("(WT)"):
                    title_val = f"{title_val} (WT)"
                    print(f"    [!] No AI Overview available. Forcing Working Title fallback: {title_val}")

            # 3. Dedicated Fallback logic
            if title_found:
                if is_definitive:
                    if parsed_metadata.get("artist_name") is None:
                        print(f"[*] Missing artist for '{title_val}'. Triggering dedicated search...")
                        artist_data = search_missing_artist(title_val, parsed_metadata)
                        if artist_data.get("artist_name"):
                            parsed_metadata["artist_name"] = artist_data["artist_name"]
                            print(f"    [+] Found Artist: {parsed_metadata['artist_name']}")

                    if parsed_metadata.get("year_of_completion") is None:
                        print(f"[*] Missing year for '{title_val}'. Triggering dedicated search...")
                        year_data = search_missing_year(title_val, parsed_metadata)
                        if year_data.get("year_of_completion"):
                            parsed_metadata["year_of_completion"] = year_data["year_of_completion"]
                            print(f"    [+] Found Year: {parsed_metadata['year_of_completion']}")
                else:
                    print(
                        "    [*] Skipping dedicated fallback searches (imageless search on a Working Title is unreliable).")

            # --- PREPARE DATA FOR SAVING ---
            parsed_metadata["original_filename"] = filepath.name
            first_name = parsed_metadata.get("artist_first_name")
            last_name = parsed_metadata.get("artist_last_name")
            if first_name and last_name:
                artist_val = f"{last_name}, {first_name}"  # Formats as "Caldwell, Clyde"
            elif last_name:
                artist_val = last_name  # Handles mononyms like "Brom"
            elif first_name:
                artist_val = first_name
            else:
                artist_val = None
            year_val = parsed_metadata.get("year_of_completion")

            # SMART NAMING LOGIC: Adjust strings based on what data actually exists
            if title_found:
                artist_str = artist_val or "Artist unknown"
                year_str = f" ({year_val})" if year_val else ""

                # Clean the LLM's title to prevent "Caldwell, Clyde - Caldwell, Clyde"
                clean_title = clean_redundant_title_text(title_val, first_name, last_name, year_val)
                # If the cleaning stripped EVERYTHING (e.g. the original title was literally just "Caldwell 1984"), fallback to something safe
                clean_title = clean_title if clean_title else "Unknown Title"

                raw_new_name = f"{artist_str} - {clean_title}{year_str}"

                # Clean strings for console output
                print_artist = artist_str
                print_title = clean_title
                print_year = year_val or "?"
            else:
                # Clean the original filename so it doesn't double-up when we prepend the artist
                clean_stem = clean_redundant_title_text(filepath.stem, first_name, last_name, year_val)
                title_str = clean_stem if clean_stem else "Unknown Image"

                artist_str = f"{artist_val} - " if artist_val else ""
                year_str = f" ({year_val})" if year_val else ""
                raw_new_name = f"{artist_str}{title_str}{year_str}"

                # Clean strings for console output
                print_artist = artist_val or ""
                print_title = title_str
                print_year = year_val or ""

            # --- RENAMING LOGIC ---
            if RENAME_FILES:
                safe_new_name = sanitize_filename(raw_new_name)
                unique_base_name = get_unique_base_name(image_dir, safe_new_name, filepath)

                new_img_path = image_dir / f"{unique_base_name}{filepath.suffix}"
                new_meta_path = image_dir / f"{unique_base_name}.json"
                new_raw_path = image_dir / f"{unique_base_name}_tmp.json"

                if new_img_path != filepath:
                    print(f"[*] Renaming files to match: {unique_base_name}")
                    filepath.rename(new_img_path)

                    # Manage the raw data file if we are saving it
                    if raw_lens_filepath.exists():
                        if SAVE_RAW_LENS_DATA:
                            raw_lens_filepath.rename(new_raw_path)
                        else:
                            # Clean up if it was left over from a previous run
                            try:
                                raw_lens_filepath.unlink()
                            except FileNotFoundError:
                                pass

                    # NEW: Clean up the old metadata file so we don't leave orphans behind
                    if metadata_filepath.exists() and metadata_filepath != new_meta_path:
                        try:
                            metadata_filepath.unlink()
                            print(f"    [*] Cleaned up old orphaned metadata: {metadata_filepath.name}")
                        except FileNotFoundError:
                            pass

                # Save parsed metadata
                with open(new_meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump(parsed_metadata, meta_file, indent=4)

                print(f"[*] Success! Saved metadata to: {new_meta_path.name}")

            else:
                # Standard Mode: Keep original names
                with open(metadata_filepath, "w", encoding="utf-8") as meta_file:
                    json.dump(parsed_metadata, meta_file, indent=4)
                print(f"[*] Success! Saved metadata to: {metadata_filepath.name}")

            # Console formatting
            if title_found:
                print(f"\n[*] Final Extracted Data:\n[==>] IDENTIFIED: {print_artist} - {print_title} ({print_year})")
            else:
                sep = " - " if print_artist else ""
                yr = f" ({print_year})" if print_year else ""
                print(f"\n[*] Final Extracted Data:\n[==>] FILE RENAMED: {print_artist}{sep}{print_title}{yr}")

        except requests.exceptions.RequestException as e:
            err_msg = f"Network/API Error on {filepath.name}: {e}"
            print(f"[!] {err_msg}")
            run_errors.append(err_msg)
        except json.JSONDecodeError as e:
            err_msg = f"JSON Truncation Error on {filepath.name}: {e}"
            print(f"[!] {err_msg}")
            run_errors.append(err_msg)
        except Exception as e:
            err_msg = f"Unexpected Error on {filepath.name}: {e}"
            print(f"[!] {err_msg}")
            run_errors.append(err_msg)

    # --- FINAL ERROR REPORTING ---
    print("\n" + "=" * 50)
    print("FINAL RUN REPORT:")
    print("=" * 50)

    if skipped_files:
        print(f"[*] Skipped {len(skipped_files)} file(s) (metadata already exists).")

    if run_errors:
        print(f"\n[*] Encountered {len(run_errors)} warning(s)/error(s):")
        for error in run_errors:
            print(f"    - {error}")

    if not run_errors:
        print("[*] Pipeline completed successfully with zero warnings or errors.")


if __name__ == "__main__":
    main()