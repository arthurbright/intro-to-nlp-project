import datasets
from datasets import load_dataset
from itertools import islice
import unicodedata
import re
from util import *

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

def download_dataset(dataset_name, column_name, num_chars):
    ds = load_dataset(
        *dataset_name,
        split="train",
        streaming=True,
        # columns=[column_name]
    )

    # data = list(islice(ds, num_entries))
    # data = [x[column_name] for x in data]

    sentences = []
    for d_ in ds:
        d = d_[column_name]
        if len(d) < 500: continue
        d = d.replace("\r", "")
        d = re.sub(r"[ \t]+", " ", d)
        d = re.sub(r"==.*?==", "", d)
        d = d.replace('\\', '')
        d = d.replace("\u200d", "")
        

        d = d.split('\n')
        d = [s for s in d if len(s) > 5]
        d = [s for s in d if '|' not in s and '{' not in s and '(' not in s and ')' not in s]
        d = [normalize(s) for s in d]


        sentences.extend(d)
        if sum([len(s) for s in sentences]) >= num_chars:
            break

    print(dataset_name, "Number of sentences:", len(sentences))
    print("Number of characters:", sum([len(s) for s in sentences]))

    # with open(output_file, "wb") as f:
    #     pickle.dump(sentences, f)
    # print(sentences)
    # print("DONE!")
    return sentences


# download_dataset(("wikimedia/wikipedia", "20231101.it"), "text", 2)
# download_dataset("VTSNLP/vietnamese_curated_dataset", "text", 3, 100000, "data/corpus/viet.pickle")
# download_dataset("Finnish-NLP/oscar_2301_fi_cleaned", "text", 4, 100000, "data/corpus/fin.pickle")
# download_dataset("musabg/wikipedia-oscar-tr", "text", 2, 100000, "data/corpus/turk.pickle")
# download_dataset(("wikimedia/wikipedia", "20231101.th"), "text", 6, 100000, "data/corpus/thai.pickle", is_thai=True)
# download_dataset(("wikimedia/wikipedia", "20231101.th"), "text", 6, 30000, "data/corpus/thai_rom.pickle", is_thai=True)
# download_dataset("somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl", "nah", 6, 100000, "data/corpus/nah.pickle")



# ds = load_dataset("wikimedia/wikipedia", "20231101.en")