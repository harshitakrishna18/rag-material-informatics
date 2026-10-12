import arxiv
from pathlib import Path

OUT = Path("data/pdfs")
OUT.mkdir(parents=True, exist_ok=True)

QUERY = '("materials informatics" OR "machine learning for materials" OR "materials discovery") AND (cat:cond-mat.mtrl-sci)'
MAX_RESULTS = 10  # Download 10 papers total

import time

# Add initial cooldown to handle previous rate limits
print("Waiting 120 seconds to cool down API rate limit...")
time.sleep(30)

client = arxiv.Client(
    page_size=5,
    delay_seconds=10.0,
    num_retries=3
)

search = arxiv.Search(
    query=QUERY,
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.Relevance,
)

max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1} to fetch results...")
        for r in client.results(search):
            # Check if file already exists to skip re-downloading
            file_name = f"{r.get_short_id()}.pdf"
            file_path = OUT / file_name
            if not file_path.exists():
                r.download_pdf(dirpath=str(OUT), filename=file_name)
                print("Downloaded", r.get_short_id(), "-", r.title[:80])
                time.sleep(2) # Extra sleep between downloads
            else:
                print("Skipping", r.get_short_id(), "- already exists")
        break # Success
    except Exception as e:
        print(f"Error encountered: {e}")
        if "429" in str(e):
            wait_time = 60 * (attempt + 1)
            print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        else:
            raise e
