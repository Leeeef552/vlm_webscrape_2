import json
import os
import argparse
import re
from typing import List, Dict
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# —————— Configuration ——————
# Make sure you have your OPENAI_API_KEY set in your environment:
#   export OPENAI_API_KEY="your_api_key_here"

# You can change the model name if needed
MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


def dedupe_meanings_via_llm(abbrev: str, meanings: List[str]) -> List[str]:
    """
    Given a list of candidate meanings, ask the LLM to return only unique senses.
    """
    system_prompt = (
        """
            You are a helpful assistant that filters a list of definitions related to the Singapore context. Use your knowledge of Singapore throughout.
            You are given an abbreviation, and a list of potential meanings; however the meanings could overlap or essentially refer to the same thing phrased differently.
            Return a JSON array of definitions that are semantically distinct (deduplicated), ensuring each item refers to a new, distinct explanation or reference.
            Ignore differences in capitalization or punctuation. Remove any entries that are mere rephrasings or hallucinations.
            Return only the cleaned list of meanings.

            Examples:
            - "MOH": ["Ministry of Health", "Ministry of Health (Singapore)"] → ["Ministry of Health"]
            - "MCCY": ["Ministry of Culture, Community and Youth", "Ministry of Culture, Community & Youth"] → ["Ministry of Culture, Community and Youth"]
            - "CSA": ["Cyber Security Agency (division under MDDI)", "Community Sentencing Act (MHA legislative framework)", 
              "Cyber Security Agency of Singapore", "Cyber Security Agency"] 
              → ["Cyber Security Agency (division under MDDI)", "Community Sentencing Act (MHA legislative framework)"]
        """
    )
    user_prompt = (
        "Here are the definitions for abbreviation '{abbr}':\n\n"
        f"{meanings}\n\n"
        "Please return a pure JSON array of the unique definitions only."
    )

    client = OpenAI(
        api_key="dummy",
        base_url="http://localhost:8124/v1"
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = resp.choices[0].message.content
    content = raw.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else ""

    # Extract JSON array block
    match = re.search(r"\[.*\]", content, flags=re.DOTALL)
    if not match:
        print(f"Error parsing LLM response for {abbrev}: no JSON array found.")
        print(f"Raw response was: {raw!r}\n")
        return meanings
    json_block = match.group(0)

    try:
        result = json.loads(json_block)
        if not isinstance(result, list):
            raise ValueError("Expected JSON list")
        return result
    except Exception as e:
        print(f"Error parsing JSON for {abbrev}: {e}")
        print(f"Extracted JSON was: {json_block!r}\nRaw response was: {raw!r}\n")
        return meanings


def dedupe_file(input_path: str, output_path: str, max_workers: int = None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data: Dict[str, List[str]] = json.load(f)

    deduped: Dict[str, List[str]] = {}
    items = list(data.items())

    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        futures = {executor.submit(dedupe_meanings_via_llm, abbr, meanings): abbr
                   for abbr, meanings in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Deduplicating"):
            abbr = futures[future]
            try:
                deduped[abbr] = future.result()
            except Exception as e:
                print(f"Error processing {abbr}: {e}")
                deduped[abbr] = data[abbr]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    print(f"\nDeduplicated file written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deduplicate meaning lists for abbreviations using an LLM"
    )
    parser.add_argument("input_json", help="Path to input JSON file (abbr → [meanings])")
    parser.add_argument("output_json", help="Path to write deduplicated JSON")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Concurrent workers (default: CPU count)")
    args = parser.parse_args()
    dedupe_file(args.input_json, args.output_json, max_workers=args.workers)
