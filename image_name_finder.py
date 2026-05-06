import os
import requests
import json
import urllib.parse
import re
import time
from pathlib import Path
from google import genai

# ==========================================
# Configuration
# ==========================================
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "????")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "????")

DIRECTORY_PATH = r"\\????\Abyss Web Server\htdocs\tmp"
BASE_URL = "https://????/tmp/"

# --- PROCESSING SWITCHES ---
RENAME_FILES = True  # Set to False to keep original filenames
TITLE_UNKNOWNS_BY_SUBJECT = True # just give it a name based on what we see if we can't find the name
SCRUB_EMOJIS_FROM_LLM_INPUT = True  # Set to True to strip emojis before sending data to Gemini (prevents some crashes)
SAVE_RAW_LENS_DATA = True  # Set to True to save the raw SerpApi response to a '_tmp.json' file
REDO_LLM = False  # Set to True to re-run Gemini on files that already have a final .json metadata file

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
ESCAPED_EMOJI_PATTERN = re.compile(r'(\\u[dD][a-fA-F0-9]{3}\\u[dD][a-fA-F0-9]{3}|\\u2[0-9a-fA-F]{3}|\\u[fF][0-9a-fA-F]{3})')

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
    "required": ["artist_name"]
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
    """Removes characters that are illegal in Windows file paths."""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return safe_name.strip()


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
    Removes search_metadata and search_parameters from the SerpApi JSON.
    This prevents 'data leakage' where the LLM sees the original filename via the query URL.
    Returns True if the data was modified.
    """
    modified = False
    for key in ["search_metadata", "search_parameters"]:
        if key in data:
            del data[key]
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
        gen_config["thinking_config"] = {"thinking_level": "low"}

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
            # 1. Handle SerpApi Data
            needs_serpapi_fetch = True

            if raw_lens_filepath.exists():
                with open(raw_lens_filepath, "r", encoding="utf-8") as f:
                    lens_data = json.load(f)

                # Scrub existing cache to prevent filename data leakage
                cache_modified = scrub_lens_data(lens_data)

                # If a local cache contains a page_token, OR if the ai_overview is literally {}, it is expired/failed.
                if "ai_overview" in lens_data and (
                        not lens_data["ai_overview"] or "page_token" in lens_data["ai_overview"]):
                    warn_msg = f"Local cache contained an expired or empty AI Overview for {filepath.name}. Discarding cache and re-fetching SerpApi."
                    print(f"    [!] {warn_msg}")
                    run_errors.append(warn_msg)
                    raw_lens_filepath.unlink()  # Delete the expired/failed cache file
                else:
                    print(f"[*] Found existing raw data: {raw_lens_filepath.name} (Skipping SerpApi)")
                    needs_serpapi_fetch = False

                    # If we modified the cache by scrubbing out the URLs, save it so it's clean for future runs
                    if cache_modified and SAVE_RAW_LENS_DATA:
                        with open(raw_lens_filepath, "w", encoding="utf-8") as f:
                            json.dump(lens_data, f, indent=4)
                        print("    [*] Scrubbed original filename leak from existing local cache.")

            if needs_serpapi_fetch:
                safe_filename = urllib.parse.quote(filepath.name)
                public_url = f"{BASE_URL}{safe_filename}"
                print(f"[*] Generated URL: {public_url}")
                print("[*] Querying SerpApi Google Lens...")

                lens_data = search_google_lens(public_url)

                # Scrub the fresh data before anything else sees it
                scrub_lens_data(lens_data)

                # Intercept and fetch the full AI Overview immediately before it expires
                if "ai_overview" in lens_data:
                    lens_data["ai_overview"] = fetch_full_ai_overview(lens_data["ai_overview"], run_errors,
                                                                      filepath.name)

                if SAVE_RAW_LENS_DATA:
                    with open(raw_lens_filepath, "w", encoding="utf-8") as json_file:
                        json.dump(lens_data, json_file, indent=4)
                    print(f"[*] Saved raw Lens data to: {raw_lens_filepath.name}")

            # Assess if the AI Overview was completely empty or failed
            missing_ai_overview = not bool(lens_data.get("ai_overview"))

            # 2. Pass the data to Gemini for extraction
            print("[*] Passing data to Gemini for analysis (Pass 1: flash-lite)...")
            gemini_output = analyze_json_with_gemini(lens_data, model_name="gemini-3.1-flash-lite-preview", temp=LENS_QUERY_TEMP)
            parsed_metadata = json.loads(gemini_output)

            # Evaluate First Pass
            title_val = parsed_metadata.get("title_of_work")
            title_found = parsed_metadata.get("title_found") and title_val
            is_definitive = title_found

            # 2b. Second Pass if Title not found
            if not title_found:
                print("[*] Title not found definitively. Retrying (Pass 2: flash)...")
                gemini_output_pass2 = analyze_json_with_gemini(lens_data, model_name="gemini-3-flash-preview", temp=LENS_QUERY_TEMP)
                parsed_metadata = json.loads(gemini_output_pass2)

                # Re-evaluate
                title_val = parsed_metadata.get("title_of_work")
                title_found = parsed_metadata.get("title_found") and title_val
                is_definitive = title_found

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
                raw_new_name = f"{artist_str} - {title_val}{year_str}"

                # Clean strings for console output
                print_artist = artist_str
                print_title = title_val
                print_year = year_val or "?"
            else:
                title_str = filepath.stem
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
