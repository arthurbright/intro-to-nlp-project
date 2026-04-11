import wikipediaapi
import re
import regex as re_u
from tqdm import tqdm
import nltk

# Download tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# -------------------------------
# CONFIG
# -------------------------------

LANGUAGES = ['en','ru','zh','ja','hi','ar','ko','fr','de','it']

USER_AGENT = "NLP-SpaceMission-Dataset/1.0 (artb1234567@example.com)"

SEED_TOPICS = {
    'en': ["Space mission", "Apollo program", "Mars exploration", "List of NASA missions"],
    'ru': ["Космическая программа", "Программа Аполлон"],
    'zh': ["太空任务", "阿波罗计划", "国际空间站", "人造衛星"],
    'ja': ["スペースシャトルのミッション一覧", "宇宙開発", "アポロ計画"],
    'hi': ["अपोलो अभियान", "अंतरिक्ष अन्वेषण", "अंतरिक्ष शटल"],
    'ar': ["الهبوط على القمر", "برنامج أبولو"],
    'ko': ["우주선", "아폴로 계획"],
    'fr': ["Mission spatiale", "Programme Apollo"],
    'de': ["Raumfahrtmission", "Apollo-Programm"],
    'it': ["Missione spaziale", "Programma Apollo"]
}

# -------------------------------
# INIT WIKIPEDIA APIS (FIXED)
# -------------------------------

WIKI_APIS = {
    lang: wikipediaapi.Wikipedia(
        language=lang,
        user_agent=USER_AGENT
    )
    for lang in LANGUAGES
}

# -------------------------------
# CLEANING
# -------------------------------

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)       # citations
    text = re.sub(r"http\S+", "", text)       # URLs
    text = re.sub(r"\s+", " ", text)          # whitespace normalize
    text = text.lower()

    # keep all language letters
    text = re_u.sub(r"[^\p{L}\s]", " ", text)

    return text.strip()


def tokenize_text(text, lang):
    no_space_langs = ['zh', 'ja', 'ko']

    # if lang in no_space_langs:
    #     tokens = [c for c in text if c.strip()]
    # else:
    #     tokens = word_tokenize(text)
    tokens = word_tokenize(text)

    return tokens


# -------------------------------
# WIKI FUNCTIONS
# -------------------------------

def get_page_text(wiki, title):
    try:
        page = wiki.page(title)
        if not page.exists():
            return None
        return page.text
    except Exception:
        return None


def expand_pages(wiki, seeds, limit=1000):
    collected = set()

    for title in seeds:
        try:
            page = wiki.page(title)
            if not page.exists():
                print("PAGE DOES NOT EXIST: ", title)
                continue

            collected.add(title)

            for link in list(page.links.keys())[:limit]:
                collected.add(link)

        except Exception:
            continue

    return list(collected)


# -------------------------------
# DATASET BUILDER
# -------------------------------

def detokenize(tokens, lang):
    no_space_langs = ['zh', 'ja', 'ko', 'hi']

    if lang in no_space_langs:
        return "".join(tokens)   # ✅ no spaces
    else:
        return " ".join(tokens)  # ✅ normal languages

def build_dataset():
    dataset = []

    for lang in LANGUAGES:
        print(f"\nProcessing: {lang}")
        wiki = WIKI_APIS[lang]

        seeds = SEED_TOPICS.get(lang, [])
        pages = expand_pages(wiki, seeds)

        for title in tqdm(pages):
            raw = get_page_text(wiki, title)
            if not raw:
                continue

            cleaned = clean_text(raw)
            tokens = tokenize_text(cleaned, lang)

            if len(tokens) < 50:
                continue

            dataset.append({
                "language": lang,
                "title": title,
                "text": detokenize(tokens, lang)
            })

    return dataset


# -------------------------------
# SAVE
# -------------------------------

def save_txt(dataset, filename="multilingual_space.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(f"{item['language']}\t{item['text']}\n")


def save_json(dataset, filename="multilingual_space.json"):
    import json
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    data = build_dataset()

    print(f"\nCollected {len(data)} documents")

    save_txt(data)
    save_json(data)

    print("Saved dataset!")