from collections import defaultdict


def str_to_cipher(str):
    assert len(str) == 26
    swaps = {}
    _res = ""
    ind = 0
    for c in 'abcdefghijklmnopqrstuvwxyz':
        swaps[str[ind]] = c
        ind += 1
    return swaps

def cipher_to_str(swaps):
    _res = ""
    for c in 'abcdefghijklmnopqrstuvwxyz':
        for x in 'abcdefghijklmnopqrstuvwxyz':
            if swaps[x].lower() == c:
                _res += x
    return _res

def sub_cipher(text: str, swaps: dict):
    res = ""
    for c in text:
        if c in swaps:
            res += swaps[c]
        elif c == ' ':
            res += " "
        else:
            res += c
    return res

def sub_cipher_dict(d, swaps):
    res = {}
    for k in d:
        res[sub_cipher(k, swaps)] = d[k]
    return res

def inv(swaps):
    res = {}
    for k in swaps:
        if swaps[k] in res:
            raise "duplicate value found inverting swaps"
        res[swaps[k]] = k
    return res


# functions given from assignment
def greedy_token_count(text: str, tokens: list[str]) -> dict:
    tokens = sorted(tokens, key=len, reverse=True)
    parsed_tokens = []

    index = 0
    count = 0
    text_len = len(text)
    counter = {}
    for t in tokens:
        counter[t] = 0

    while index < text_len:
        matched = False
        for token in tokens:
            if text.startswith(token, index):
                index += len(token)
                matched = True
                counter[token] += 1
                parsed_tokens.append(token)
                break
        if not matched:
            raise Exception(f'Failed to tokenize at index {index} in text: {text}')
        count += 1
    return count, counter, parsed_tokens

def compression_ratio(tokens, file) -> float:    
    tokens = sorted(tokens, key=len, reverse=True)
    total_chars = 0
    total_tokens = 0
    counter = defaultdict(int)
    with open(file) as fin:
        for line in fin.readlines(): 
            line = line.rstrip('\n')
            total_chars += len(line)
            cc, counter_, _ = greedy_token_count(line, tokens)
            total_tokens += cc
            for word in counter_:
                counter[word] += counter_[word]
    compression_ratio = total_chars / float(total_tokens)
    return compression_ratio, counter

#########
START = 0
END = 1

ENCODING = 'utf-8'


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))

    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = { BYTE_TO_CHAR[b]:b for b in BYTE_TO_CHAR}


def TO_BYTES(s: str):
    b = s.encode(ENCODING)
    return "".join([BYTE_TO_CHAR[byte] for byte in b])

def FROM_BYTES(b):
    b = bytes([CHAR_TO_BYTE[byte] for byte in b])
    return b.decode(ENCODING)



def print_bytes(b):
    try:
        print(b.decode(ENCODING))
    except:
        print(b)

def bytes_type(c: str):
    """ 1-4: starting byte of a char
  
    -x: x middle chars, followed by a start of char
    10 + x: x middle chars
    
    """
    b = CHAR_TO_BYTE[c[0]]
    if b < 128:
        return 1
    if b < 128 + 64:
        cnt = 0
        for char in c:
            if 128 <= CHAR_TO_BYTE[char] < 128 + 64:
                cnt += 1
            else:
                return -cnt
        return 10 + cnt
    if b < 128 + 64 + 32:
        return 2
    if b < 128 + 64 + 32 + 16:
        return 3
    return 4

def num_trailing_middle_bytes_needed(s: str):
    acc = 0
    for c in reversed(s):
        t = bytes_type(c)
        if t == 0:
            acc -= 1
        else:
            return t + acc
    raise "trailing muiddle bytes calculation: no start of char found"


def escape_csv(line):
    return f'"{line.replace('"', '""')}"'