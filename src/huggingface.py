import datasets
from datasets import load_dataset
import unicodedata
import re
import spacy
import pickle
from util import *

THAI_MAP = str.maketrans({
    # Consonants
    "ก": "k", "ข": "kh", "ฃ": "kh", "ค": "kh", "ฅ": "kh", "ฆ": "kh",
    "ง": "ng", "จ": "ch", "ฉ": "ch", "ช": "ch", "ซ": "s", "ฌ": "ch",
    "ญ": "y", "ฎ": "d", "ฏ": "t", "ฐ": "th", "ฑ": "th", "ฒ": "th",
    "ณ": "n", "ด": "d", "ต": "t", "ถ": "th", "ท": "th", "ธ": "th",
    "น": "n", "บ": "b", "ป": "p", "ผ": "ph", "ฝ": "f", "พ": "ph", "ฟ": "f",
    "ภ": "ph", "ม": "m", "ย": "y", "ร": "r", "ฤ": "rue", "ล": "l", "ฦ": "lue",
    "ว": "w", "ศ": "s", "ษ": "s", "ส": "s", "ห": "h", "ฬ": "l", "อ": "o", "ฮ": "h",

    # Vowels (short/long)
    "ะ": "a", "ั": "a", "า": "a", "ิ": "i", "ี": "i", "ึ": "ue", "ื": "ue",
    "ุ": "u", "ู": "u", "เ": "e", "แ": "ae", "โ": "o", "ใ": "ai", "ไ": "ai",
    "ๅ": "a", "ๆ": "", "็": "", "์": "", "ํ": "m", "๎": "",
    "ำ": "am",
    "æ": "ae",
    "ฯ": "",

    # Tone marks
    "่": "", "้": "", "๊": "", "๋": "",

    # Numbers (optional)
    "๐": "0", "๑": "1", "๒": "2", "๓": "3", "๔": "4",
    "๕": "5", "๖": "6", "๗": "7", "๘": "8", "๙": "9",

    # Punctuation (optional, map to English equivalents)
    "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "…": "...",
})

def thai_to_english_lower(text: str) -> str:
    try:
        res = romanize(text, engine="tltk").lower()
    except:
        res = ""
    # print(text, res)
    return res

def fast_thai_to_english(text: str) -> str:
    return text.translate(THAI_MAP).lower()



nlp = spacy.blank("hu")
nlp.add_pipe("sentencizer")

def remove_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    res = "".join(
        ch for ch in normalized
        if unicodedata.category(ch) != "Mn"
    )
    res = res.replace("đ", "d").replace("Đ", "D")
    res = res.replace("ı", "i")
    return res

def split_sentences(text: str):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def clean_sentence(text: str):
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = text.replace('\n', '').replace('\r', '').replace('\t', '')
    return text.strip()

_DIGIT_RE = re.compile(r"\d")

def contains_digits(sentence: str) -> bool:
    return bool(_DIGIT_RE.search(sentence))


pattern = re.compile(r'[A-Za-z]')
def filter_no_english(strings):
    """
    Keep only strings that do NOT contain English letters (a-z, A-Z)
    """
    return [s for s in strings if not pattern.search(s)]

VALID_CHARS = set([c for c in 'abcdefghijklmnopqrstuvwxyz '])
def valid_sentence(text: str):
    chars = set([c for c in text])
    res =  chars.issubset(VALID_CHARS)
    if not res:
        print("INVALID SENTENCE", text)
        print(chars - VALID_CHARS)
        # quit()
    return res

#########################################
# HUNGARIAN

def download_dataset(dataset_name, column_name, lang_id, num_sentences, output_file, is_thai = False):
    ds = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        # columns=[column_name]
    )

    # data = list(islice(ds, num_entries))
    # data = [x[column_name] for x in data]

    sentences = []
    for d_ in ds:
        d = d_[column_name]
        if is_thai:
            d = d.split() # split by space
            d = [s for s in d if filter_no_english(s)]
            d = [thai_to_english_lower(s) for s in d]
            d = [s for s in d if len(s) > 2]
        else:
            d = split_sentences(d)

        d = [remove_diacritics(s) for s in d]
        d = [s.lower() for s in d]
        d = [clean_sentence(s) for s in d]  # remove grammar
        d = [s for s in d if not contains_digits(s)] # remove digits
        d = [s for s in d if valid_sentence(s)]  # double check
        sentences += d
        if len(sentences) >= num_sentences:
            break

    # upsample original data
    # cipher = str_to_cipher('ljzkthwmcqsxfgounpdebivyra')
    # for i in range(5):
    #     with open(f"lang_0{lang_id}.txt") as f:
    #         for line in f:
    #             line = line.strip()
    #             sentences.append(sub_cipher(line, cipher))

    print("Number of sentences:", len(sentences))
    print("Number of characters:", sum([len(s) for s in sentences]))

    with open(output_file, "wb") as f:
        pickle.dump(sentences, f)
    print("DONE!")


# download_dataset("SZTAKI-HLT/HunSum-1", "article", 1, 100000, "data/corpus/hungarian20.pickle")
# download_dataset("VTSNLP/vietnamese_curated_dataset", "text", 3, 100000, "data/corpus/viet.pickle")
# download_dataset("Finnish-NLP/oscar_2301_fi_cleaned", "text", 4, 100000, "data/corpus/fin.pickle")
# download_dataset("musabg/wikipedia-oscar-tr", "text", 2, 100000, "data/corpus/turk.pickle")
# download_dataset(("wikimedia/wikipedia", "20231101.th"), "text", 6, 100000, "data/corpus/thai.pickle", is_thai=True)
# download_dataset(("wikimedia/wikipedia", "20231101.th"), "text", 6, 30000, "data/corpus/thai_rom.pickle", is_thai=True)
# download_dataset("somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl", "nah", 6, 100000, "data/corpus/nah.pickle")


###### trs files
import xml.etree.ElementTree as ET

def extract_trs_text(filepath: str) -> str:
    # IMPORTANT: open with correct encoding
    with open(filepath, "r", encoding="ISO-8859-1") as f:
        tree = ET.parse(f)

    root = tree.getroot()
    texts = []

    for turn in root.findall(".//Turn"):
        # Text directly inside <Turn> before first child
        if turn.text:
            texts.append(turn.text.strip())

        for child in turn:
            # Text that appears after tags like <Sync/> or <Comment/>
            if child.tail:
                texts.append(child.tail.strip())

    # Clean empty strings and join
    res = (t.replace(":", "").replace("*", "").replace('³', "") for t in texts if t)
    return res


import os

def get_files(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

# Example
nah_files = get_files("Zacatlan-Tepetzintla-Nahuatl-Transcriptions/current")

nah_sentences = []
for file in nah_files:
    d = extract_trs_text(file)

    d = [remove_diacritics(s) for s in d]
    d = [s.lower() for s in d]
    d = [clean_sentence(s) for s in d]  # remove grammar
    d = [s for s in d if not contains_digits(s)] # remove digits
    d = [s for s in d if valid_sentence(s)]  # double check
    nah_sentences += d
    if len(nah_sentences) >= 100000:
        break

print("Number of sentences:", len(nah_sentences))
print("Number of characters:", sum([len(s) for s in nah_sentences]))

with open("data/corpus/nah.pickle", "wb") as f:
    pickle.dump(nah_sentences, f)
print("DONE!")

    
