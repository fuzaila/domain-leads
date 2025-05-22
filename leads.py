import os, json, time, hashlib
from pathlib import Path

import requests
import pandas as pd
import streamlit as st
import tldextract, wordninja # Ensure wordninja is installed: pip install wordninja
import google.generativeai as genai
from dotenv import load_dotenv

# ── 0. Load keys ─────────────────────────────────────────────────────────
load_dotenv()
SERP_KEY   = os.getenv("SERPAPI_KEY")    or st.secrets.get("SERPAPI_KEY", "")
GEMINI_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GEMINI_KEY", "")

if not (SERP_KEY and GEMINI_KEY):
    st.error("Please set SERPAPI_KEY & GOOGLE_API_KEY in .env or secrets.toml")
    st.stop()

# ── 1. Configure Gemini ───────────────────────────────────────────────────
genai.configure(api_key=GEMINI_KEY)
GEM_MODEL = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash-latest", # Or your preferred working model
    generation_config={"response_mime_type": "application/json"}
)

# ── 2. Global consts & cache ──────────────────────────────────────────────
QUERY_NUM   = 100
CHUNK_SIZE  = 12 # Keep this manageable for Gemini's context and your patience
CACHE_DIR   = Path("serp_cache")
CACHE_DIR.mkdir(exist_ok=True)
SYS_PROMPT  = """
Return STRICT JSON array. Each item:
  { "url": "...", "score": <0–10>, "reason": "..." }
• 9–10: perfect buyer  • 6–8: strong thematic fit  • ≤5: omit
Discard directories, socials, YouTube, Facebook, Pinterest, etc.
"""

# ── 3. Helper functions ────────────────────────────────────────────────────
def get_quota():
    try:
        j = requests.get("https://serpapi.com/account.json",
                         params={"api_key":SERP_KEY}, timeout=10).json()
        return j.get("plan_searches_left", 0), j.get("searches_per_month", 0)
    except requests.RequestException as e:
        st.warning(f"Could not fetch SerpAPI quota: {e}")
        return 0, 0

# Define a list of TLDs that should be considered as keywords
KEYWORD_TLDS = {"ai", "io", "bot", "co", "info", "app", "dev", "tech", "store", "design",
                "art", "photo", "pics", "studio", "digital", "media", "news",
                "marketing", "agency", "solutions", "systems", "cloud", "data",
                "finance", "money", "cash", "invest", "capital", "trading",
                "games", "play", "bet", "casino",
                "health", "care", "clinic", "doctor", "dental", "vet",
                "life", "bio", "eco", "green",
                "travel", "tours", "holiday", "vacation", "place", "world",
                "shop", "sale", "online", "market",
                "auto", "car", "cars", "moto",
                "house", "home", "homes", "realty", "estate", "villas", "villa", # Added villa/villas
                "build", "construction", "contractors",
                "law", "legal", "attorney",
                "school", "college", "university", "academy", "degree", "courses", "study",
                "band", "music", "audio", "live", "show",
                "charity", "foundation", "gives",
                "food", "restaurant", "cafe", "bar", "pizza", "delivery",
                "fashion", "style", "beauty", "hair", "skin",
                "fit", "fitness", "gym", "yoga",
                "apartments", "rent", "rentals", "condos",
                "space", "xyz", "site", "website", "click", "link", "fyi"} # Add more as you see fit

BRAND_SLD_MAX_LEN = 7 # Adjust as needed. "blixy" is 5. "company" is 7. "solution" is 8.
MIN_WORDNINJA_WORD_LEN = 3 # If wordninja splits into words shorter than this, distrust it.

def extract_keywords(domain: str) -> str:
    """
    Extracts keywords from a domain.
    - Tries to split SLD using wordninja if it seems like a compound of actual words.
    - Keeps SLD intact if it looks like a brand name or coined term.
    - Appends semantically relevant TLDs as keywords.
    """
    try:
        ext_result = tldextract.TLDExtract(cache_dir=None)(domain)
        sld = ext_result.domain
        tld_suffix = ext_result.suffix.lower()

        if not sld: # Handle cases where domain might be just TLD or invalid
            if tld_suffix in KEYWORD_TLDS: return tld_suffix
            return domain

        base_keywords_list = []
        wordninja_split = wordninja.split(sld)

        # Heuristic to decide whether to use wordninja's split or the original SLD:
        # 1. If wordninja doesn't split it (returns the sld as a single word), use the sld.
        # 2. If wordninja splits it into multiple words:
        #    a. And all (or most) of those words are reasonably long (>= MIN_WORDNINJA_WORD_LEN), trust wordninja.
        #    b. If wordninja splits a short SLD (<= BRAND_SLD_MAX_LEN) into very short pieces,
        #       it's likely misinterpreting a brand name. In this case, use the original SLD.
        #    c. If wordninja splits a longer SLD, it's more likely a true compound.

        if len(wordninja_split) == 1 and wordninja_split[0] == sld:
            # Wordninja didn't split it, or it's a single dictionary word. Use the SLD.
            base_keywords_list = [sld]
        elif len(wordninja_split) > 1:
            # Wordninja did split it. Apply heuristics.
            all_words_long_enough = all(len(word) >= MIN_WORDNINJA_WORD_LEN for word in wordninja_split)
            
            if len(sld) <= BRAND_SLD_MAX_LEN and not all_words_long_enough:
                # Short SLD split into tiny pieces by wordninja - likely a brand. Use original SLD.
                base_keywords_list = [sld]
            else:
                # Either a longer SLD split, or a short SLD split into decent words. Trust wordninja.
                base_keywords_list = wordninja_split
        else: # Should not happen if sld is not empty, but as a fallback
            base_keywords_list = [sld]


        # Process TLD
        final_keywords_list = list(base_keywords_list) # Make a copy
        
        # Check the primary part of the TLD (e.g., 'ai' from 'company.ai', 'co' from 'my.business.co.uk')
        actual_tld_part_to_check = tld_suffix.split('.')[0]

        if actual_tld_part_to_check in KEYWORD_TLDS:
            # Add TLD if it's a keyword TLD and not already the last part of the base keywords
            if not final_keywords_list or actual_tld_part_to_check.lower() != final_keywords_list[-1].lower():
                final_keywords_list.append(actual_tld_part_to_check)

        # Join and remove duplicates while preserving order
        seen = set()
        unique_ordered_keywords = [x for x in final_keywords_list if x and not (x in seen or seen.add(x))] # Added 'if x' to filter empty strings

        return " ".join(unique_ordered_keywords).strip()

    except Exception as e:
        st.warning(f"Error in extract_keywords for {domain}: {e}. Using domain as keyword.")
        return domain


def cache_key(query: str, location: str|None) -> str:
    return hashlib.sha1(f"{query}||{location or ''}".encode()).hexdigest()

def fetch_serp(query: str, location: str|None) -> list[dict]:
    key = cache_key(query, location)
    fpath = CACHE_DIR / f"{key}.json"
    if fpath.exists():
        try:
            # Add a try-except for potentially corrupted cache files
            cached_data = json.loads(fpath.read_text())
            if "organic_results" in cached_data: # Ensure the key exists
                 return cached_data["organic_results"]
            else: # Cache file is not in expected format
                st.warning(f"Cache file {fpath} missing 'organic_results'. Re-fetching.")
        except json.JSONDecodeError:
            st.warning(f"Corrupted cache file {fpath}. Re-fetching.")
            # Optionally delete corrupted cache: os.remove(fpath)


    params = {"engine":"google","q":query,"num":QUERY_NUM,
              "api_key":SERP_KEY,"hl":"en","gl":"us"}
    if location: params["location"] = location

    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=30)
        resp.raise_for_status() # Raise an exception for bad status codes
        resp_json = resp.json()
        data = resp_json.get("organic_results", [])
        # Cache even if data is empty, to avoid re-querying for no results
        fpath.write_text(json.dumps({"organic_results":data}))
        return data
    except requests.RequestException as e:
        st.error(f"SerpAPI request failed for query '{query}': {e}")
        return []
    except json.JSONDecodeError:
        st.error(f"SerpAPI response for query '{query}' was not valid JSON.")
        return []


def score_chunk(domain_being_analyzed: str, items: list[dict]) -> list[dict]:
    if not items:
        return []
    lite = [{"title": i.get("title", ""), "url": i.get("link", "")} for i in items]

    full_prompt_content = (
        f"You are an expert lead scorer. The user is trying to find potential buyers "
        f"for the domain name: '{domain_being_analyzed}'.\n\n"
        f"Analyze the following Google Search Results (SERP slice) and score each URL "
        f"based on how good a fit it would be as a potential buyer for '{domain_being_analyzed}'.\n"
        f"SERP slice:\n{json.dumps(lite, indent=2)}\n\n"
        f"Use the following scoring criteria:\n"
        f"• 9–10: Perfect buyer - the company/site is in the exact niche and seems like an ideal match.\n"
        f"• 6–8: Strong thematic fit - the company/site is in a closely related niche or could clearly benefit from this domain.\n"
        f"• ≤5: Omit - not a good fit, or it's a directory, social media (YouTube, Facebook, Pinterest, LinkedIn, Twitter/X, Instagram), news aggregator, or other non-target site.\n\n"
        f"Return your analysis as a STRICT JSON array. Each item in the array must be an object with these exact keys: "
        f"{{ \"url\": \"...\", \"score\": <integer 0-10>, \"reason\": \"brief justification...\" }}\n"
        f"Only include items with a score greater than 5."
    )

    try:
        raw = GEM_MODEL.generate_content(
            full_prompt_content,
            safety_settings={
                "HARASSMENT": "block_none", "HATE": "block_none",
                "SEXUAL": "block_none", "DANGEROUS": "block_none",
            }
        )

        # --- Debugging output ---
        debug_container = st.expander(f"Gemini Debug for: {domain_being_analyzed} (chunk starting with {lite[0]['url'] if lite else 'N/A'})", expanded=False)
        with debug_container:
            st.write("Gemini Raw Prompt (first 500 chars):")
            st.code(full_prompt_content[:500] + "...")
            st.write("Gemini Raw Response (first 800 chars):")
            st.code(raw.text[:800] + ("..." if len(raw.text) > 800 else ""))
        # --- End Debugging ---

        txt = raw.text.strip()
        # More robust JSON extraction: look for the first '[' and last ']'
        start_index = txt.find('[')
        end_index = txt.rfind(']')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = txt[start_index : end_index + 1]
            try:
                parsed_json = json.loads(json_str)
                # Validate structure of parsed_json items if necessary
                valid_results = []
                for item in parsed_json:
                    if isinstance(item, dict) and "url" in item and "score" in item and "reason" in item:
                        try:
                            item["score"] = int(item["score"]) # Ensure score is an int
                            valid_results.append(item)
                        except ValueError:
                            with debug_container: st.warning(f"Invalid score format in item: {item}. Skipping.")
                    else:
                        with debug_container: st.warning(f"Invalid item structure in Gemini response: {item}. Skipping.")
                
                with debug_container: st.success(f"Successfully parsed {len(valid_results)} valid JSON objects.")
                return valid_results
            except json.JSONDecodeError as e:
                with debug_container: st.warning(f"JSONDecodeError: {e}. Extracted text was: {json_str}")
                return []
        else:
            with debug_container: st.warning(f"Could not find JSON array in Gemini response. Raw text: {txt}")
            return []

    except Exception as e:
        st.error(f"Exception in score_chunk for domain {domain_being_analyzed}: {e}")
        # if hasattr(raw, 'prompt_feedback'): st.write(raw.prompt_feedback)
        return []


def get_leads(domain_to_analyze: str, location: str | None) -> pd.DataFrame:
    st.write(f"Processing domain: {domain_to_analyze}") # Log current domain
    kw = extract_keywords(domain_to_analyze)
    st.write(f"Keywords for {domain_to_analyze}: '{kw}'")

    results = fetch_serp(kw, location)
    st.write(f"Fetched {len(results)} SERP results for '{kw}'.")

    if not results:
        st.write(f"No SERP results for {domain_to_analyze}, skipping scoring.")
        return pd.DataFrame() # Return empty DataFrame if no SERP results

    all_rows = []
    for i in range(0, len(results), CHUNK_SIZE):
        chunk = results[i:i+CHUNK_SIZE]
        st.write(f"Scoring chunk {i//CHUNK_SIZE + 1} for {domain_to_analyze} (size: {len(chunk)})")
        scored_items = score_chunk(domain_to_analyze, chunk)
        all_rows.extend(scored_items)
        if len(results) > CHUNK_SIZE : # Only sleep if multiple chunks
             time.sleep(1.1) # Slightly more than 1s for 60QPM, to be safe

    if not all_rows:
        st.write(f"No leads scored by Gemini for {domain_to_analyze}.")
        return pd.DataFrame()

    # Merge, keeping the highest score for any given URL
    merged = {}
    for r in all_rows:
        # Basic validation (already done more thoroughly in score_chunk, but good as a safeguard)
        if not isinstance(r, dict) or "url" not in r or "score" not in r:
            continue
        u = r["url"]
        current_score = r.get("score", 0)
        if u not in merged or current_score > merged[u].get("score", 0):
            merged[u] = r
    
    df = pd.DataFrame(list(merged.values())) # Convert values view to list for DataFrame

    if not df.empty:
        df = df.sort_values("score", ascending=False)
        # Add the input domain to each lead row FOR THIS BATCH
        df["input_domain"] = domain_to_analyze
        st.write(f"Found {len(df)} potential leads for {domain_to_analyze} after merging.")
    else:
        st.write(f"No leads found for {domain_to_analyze} after merging.")

    return df


# ── 4. Streamlit UI ────────────────────────────────────────────────────────
st.set_page_config("Bulk Lead Finder", "🚀", layout="wide")
st.title("🚀 Bulk Domain Lead Finder")
st.markdown("Find potential buyers for your domains by analyzing Google Search Results with AI.")

# Step 1: input & preview
domains_input_str = st.text_area("Enter Domains (one per line)", height=180,
    placeholder="example.com\ncompany.ai\nluxury-villas.net")
location_input = st.text_input("Optional: Google Search Location",
                             placeholder="e.g., Chicago, Illinois, United States or leave blank for global")

# Display quota
try:
    credits_left, credits_total = get_quota()
    st.sidebar.metric("SerpAPI Credits Left", f"{credits_left}/{credits_total}")
except Exception as e:
    st.sidebar.error(f"Could not fetch SerpAPI quota: {e}")
    credits_left = 0 # Assume 0 if quota check fails, to prevent accidental usage

if st.button("🔍 Preview Keywords & Estimated Cost (no credits used yet)"):
    domains_list = [d.strip().lower() for d in domains_input_str.splitlines() if d.strip()]
    if not domains_list:
        st.warning("Please enter at least one domain.")
    else:
        preview_data = []
        estimated_new_searches = 0
        for domain_item in domains_list:
            keywords = extract_keywords(domain_item)
            preview_data.append({"domain": domain_item, "keywords": keywords})
            
            # Check cache for cost estimation
            cache_file_path = CACHE_DIR / f"{cache_key(keywords, location_input.strip() or None)}.json"
            if not cache_file_path.exists():
                estimated_new_searches += 1
        
        df_preview = pd.DataFrame(preview_data)
        st.subheader("Keyword Preview")
        st.dataframe(df_preview, use_container_width=True)
        st.info(f"Previewed {len(domains_list)} domains. Estimated SerpAPI credits needed for new searches: {estimated_new_searches}.")
        st.caption("This does not use any SerpAPI credits. Actual credits will be consumed when you Fetch & Score.")

# Step 2: confirm & run
if st.button("▶️ Fetch & Score Leads"):
    domains_to_process = [d.strip().lower() for d in domains_input_str.splitlines() if d.strip()]
    final_location = location_input.strip() or None

    if not domains_to_process:
        st.warning("Please enter at least one domain."); st.stop()

    # Re-calculate estimated cost and check credits again before actual run
    current_credits_left_before_run, _ = get_quota()
    actual_credits_to_be_used = 0
    for domain_item_check in domains_to_process:
        keywords_check = extract_keywords(domain_item_check)
        cache_file_path_check = CACHE_DIR / f"{cache_key(keywords_check, final_location)}.json"
        if not cache_file_path_check.exists():
            actual_credits_to_be_used += 1

    if actual_credits_to_be_used > current_credits_left_before_run:
        st.error(f"You need {actual_credits_to_be_used} SerpAPI credits for new searches, but you only have {current_credits_left_before_run} left. Please reduce the number of domains or top up your SerpAPI credits."); st.stop()
    
    if actual_credits_to_be_used > 0:
        st.info(f"This will use approximately {actual_credits_to_be_used} SerpAPI credits for new searches.")
    else:
        st.info("All domain keywords seem to be cached. No new SerpAPI credits should be used.")


    all_leads_dfs = []
    run_summary_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_domains_to_process = len(domains_to_process)

    overall_start_time = time.time()

    # --- Main Processing Loop ---
    for index, current_domain in enumerate(domains_to_process):
        status_text.text(f"Processing domain {index + 1}/{total_domains_to_process}: {current_domain}...")
        domain_start_time = time.time()

        # get_leads will now return a DataFrame with 'input_domain' column
        leads_df_for_domain = get_leads(current_domain, final_location)
        
        domain_processing_time = time.time() - domain_start_time
        
        # For summary, get SERP hits (will use cache if already fetched by get_leads)
        keywords_for_summary = extract_keywords(current_domain)
        serp_results_for_summary = fetch_serp(keywords_for_summary, final_location)

        run_summary_data.append({
            "input_domain": current_domain,
            "keywords_used": keywords_for_summary,
            "serp_results_count": len(serp_results_for_summary),
            "leads_found": len(leads_df_for_domain),
            "time_taken_sec": round(domain_processing_time, 2)
        })

        if not leads_df_for_domain.empty:
            all_leads_dfs.append(leads_df_for_domain)
        
        progress_bar.progress((index + 1) / total_domains_to_process)
    # --- End Main Processing Loop ---
    
    overall_processing_time = time.time() - overall_start_time
    status_text.success(f"All domains processed in {round(overall_processing_time, 2)} seconds!")

    st.subheader("📊 Run Summary")
    summary_df = pd.DataFrame(run_summary_data)
    st.dataframe(summary_df, use_container_width=True)

    if all_leads_dfs:
        combined_leads_df = pd.concat(all_leads_dfs, ignore_index=True)
        
        # Reorder columns to put 'input_domain' first
        if "input_domain" in combined_leads_df.columns:
            cols = ["input_domain"] + [col for col in combined_leads_df.columns if col != "input_domain"]
            combined_leads_df = combined_leads_df[cols]
        
        st.subheader("✅ Combined Leads Found")
        st.dataframe(combined_leads_df, use_container_width=True)
        
        csv_data = combined_leads_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Leads as CSV",
            data=csv_data,
            file_name="bulk_domain_leads.csv",
            mime="text/csv",
        )
    else:
        st.info("No leads were found across all processed domains.")

# Sidebar with some help text
st.sidebar.header("ℹ️ How To Use")
st.sidebar.markdown("""
1.  **Enter Domains:** List one domain per line that you want to find leads for.
2.  **Optional Location:** Specify a Google search location (e.g., "New York, USA") to target leads geographically. Leave blank for global results.
3.  **Preview Keywords:** Click to see the keywords that will be extracted for each domain and estimate SerpAPI credit usage. This step is free.
4.  **Fetch & Score Leads:** Click this to start the process. SerpAPI credits will be used for domains not found in the local cache. Results will be scored by Gemini.
5.  **Review & Download:** Check the summary and the combined leads table. Download your leads as a CSV.
""")
st.sidebar.header("⚠️ Notes")
st.sidebar.markdown("""
- **Caching:** SerpAPI results are cached locally in `serp_cache` to save credits on re-runs.
- **Gemini Rate Limits:** A delay is added between processing chunks for Gemini to respect free-tier API limits (approx. 60 queries per minute).
- **Debugging:** Expand the "Gemini Debug" sections that appear during processing to see raw AI responses if you encounter issues.
""")
