import requests
from datasets import load_dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

LANGS = ['en','ru','zh','ja','hi','ar','ko','fr','de','it']

HEADERS = {"User-Agent": "fast-space-pipeline/1.0"}


# -------------------------------------------------
# SAFE REQUEST
# -------------------------------------------------
def req(url, params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    if r.status_code != 200 or "<html" in r.text.lower():
        return None
    return r.json()


# -------------------------------------------------
# STEP 1: GET LARGE SPACE MISSION SET (WIKIDATA)
# -------------------------------------------------
def get_space_missions(limit=5000):
    url = "https://query.wikidata.org/sparql"

    query = f"""
    SELECT DISTINCT ?item ?itemLabel WHERE {{

      # --------------------------------------------------
      # 1. STRUCTURED SPACE ENTITIES
      # --------------------------------------------------
      VALUES ?types {{
        wd:Q2133344   # space mission
      }}
      {{
        ?item (wdt:P31|wdt:P279) ?types .
      }}
      UNION

      # --------------------------------------------------
      # 2. TOPICAL SPACE TRAVEL ARTICLES
      # --------------------------------------------------
      {{
        ?item wdt:P921 ?topic .
        ?topic (wdt:P31|wdt:P279)* wd:Q5916 .   # spaceflight topic
      }}

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
      }}
    }}
    LIMIT {limit}
    """

    headers = {
        "User-Agent": "space-expander/1.0",
        "Accept": "application/sparql-results+json"
    }

    r = requests.get(url, params={"query": query}, headers=headers)

    if r.status_code != 200:
        raise RuntimeError(f"SPARQL failed: {r.status_code}")

    data = r.json()

    titles = []
    for row in data["results"]["bindings"]:
        titles.append(row["itemLabel"]["value"])

    return list(set(titles))


# -------------------------------------------------
# STEP 2: TITLE → QID
# -------------------------------------------------
def get_qids(titles):
    url = "https://en.wikipedia.org/w/api.php"
    out = {}

    for i in range(0, len(titles), 50):
        batch = titles[i:i+50]

        params = {
            "action": "query",
            "prop": "pageprops",
            "titles": "|".join(batch),
            "format": "json"
        }

        data = req(url, params)
        if not data:
            continue

        for page in data["query"]["pages"].values():
            if "pageprops" in page:
                out[page["title"]] = page["pageprops"]["wikibase_item"]

    return out


# -------------------------------------------------
# STEP 3: WIKIDATA SITELINKS (FAST BULK)
# -------------------------------------------------
def get_sitelinks(qids):
    url = "https://www.wikidata.org/w/api.php"
    res = {}

    for i in range(0, len(qids), 50):
        batch = qids[i:i+50]

        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "sitelinks",
            "format": "json"
        }

        data = req(url, params)
        if not data:
            continue

        for qid, ent in data["entities"].items():
            langs = {}
            for l in LANGS:
                k = f"{l}wiki"
                if k in ent.get("sitelinks", {}):
                    langs[l] = ent["sitelinks"][k]["title"]
            res[qid] = langs

    return res


# -------------------------------------------------
# STEP 4: BUILD HF INDEX (CRITICAL SPEEDUP)
# -------------------------------------------------
def build_hf_index(lang):
    print(f"Indexing {lang} Wikipedia...")

    ds = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang}",
        split="train",
        streaming=True
    )

    index = {}
    for row in ds:
        index[row["title"]] = row["text"]

    return lang, index


# -------------------------------------------------
# STEP 5: PARALLEL INDEX BUILD
# -------------------------------------------------
def build_all_indices():
    indices = {}

    with ThreadPoolExecutor(max_workers=len(LANGS)) as ex:
        futures = [ex.submit(build_hf_index, l) for l in LANGS]

        for f in futures:
            lang, idx = f.result()
            indices[lang] = idx

    return indices


# -------------------------------------------------
# STEP 6: BUILD FINAL DATASET (O(1) LOOKUP)
# -------------------------------------------------
def build_dataset(N=200):
    print("1. Fetch space missions...")
    titles = get_space_missions(N)
    print(titles)
    print(len(titles))

    print("2. Get QIDs...")
    title_to_qid = get_qids(titles)
    qids = list(title_to_qid.values())

    print("3. Get sitelinks...")
    qid_to_titles = get_sitelinks(qids)

    print("4. Build HF indices (FAST)...")
    hf_index = build_all_indices()

    print("5. Assemble dataset...")

    data = []

    for qid, langs in qid_to_titles.items():
        entry = {"qid": qid, "texts": {}}

        for lang, title in langs.items():
            if lang in hf_index and title in hf_index[lang]:
                entry["texts"][lang] = hf_index[lang][title]

        if entry["texts"]:
            data.append(entry)

    return data


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    data = build_dataset(100_000)

    print("TOTAL:", len(data))
    print(data[0])

    all_texts = [
        text
        for item in data
        for text in item["texts"].values()
    ]

    import pickle
    with open("wikidata.pickle", 'wb') as f:
        pickle.dump(all_texts, f)