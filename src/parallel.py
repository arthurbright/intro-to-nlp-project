import torch.multiprocessing as mp
import torch
from lstm import *
import pickle
import torch.nn.functional as F


torch.set_num_threads(8)

model = None
char2idx = None
idx2char = None
vocab_size = None
device = "cpu"

def init_worker(work_dir):
    global model, char2idx, idx2char, vocab_size

    with open(f"{work_dir}/params.pickle", 'rb') as f:
        params = pickle.load(f)
    vocab_size = params["vocab_size"]
    char2idx = params["char2idx"]
    idx2char = params["idx2char"]
    
    # Load model inside each process
    model = CharLSTM(vocab_size)
    model.load_state_dict(torch.load(f"{work_dir}/model.pth", weights_only=True))
    # model = torch.quantization.quantize_dynamic(
    #     model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    # )
    model.eval()

def predict_top3(text, _idx, device="cpu", k=3):
        global model, char2idx, idx2char, vocab_size

        model.eval()
        hidden = None

        # Feed the full string to build state
        for ch in text:
            if ch not in char2idx:
                ch = " "  # fallback for unknown chars

            idx = torch.tensor([[char2idx[ch]]]).to(device)

            with torch.no_grad():
                output, hidden = model(idx, hidden)

        # Get prediction for next character
        logits = output[0, -1]
        probs = torch.softmax(logits, dim=0)

        top_probs, top_indices = torch.topk(probs, k)

        results = []
        for p, idx in zip(top_probs, top_indices):
            # results.append((
            #     idx2char[idx.item()],
            #     p.item()
            # ))
            results.append(idx2char[idx.item()])

        if(_idx % 100 == 0):
            print("Done idx", _idx)
        return results

def predict_top3_batch(texts, device="cpu", k=3):
    global model, char2idx, idx2char, vocab_size
    model.eval()

    # Convert to indices
    encoded = []
    lengths = []

    for text in texts:
        seq = [
            char2idx.get(ch, char2idx[" "])
            for ch in text
        ]
        encoded.append(seq)
        lengths.append(len(seq))

    max_len = max(lengths)

    # Pad sequences
    padded = []
    for seq in encoded:
        padded.append(seq + [0] * (max_len - len(seq)))

    x = torch.tensor(padded, dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        output, _ = model(x)

    results = []

    for i, length in enumerate(lengths):
        logits = output[i, length - 1]  # last valid timestep
        probs = F.softmax(logits, dim=0)

        top_probs, top_indices = torch.topk(probs, k)

        preds = [
            # (idx2char[idx.item()], prob.item())
            # for idx, prob in zip(top_indices, top_probs)
            idx2char[idx.item()] for idx in top_indices
        ]

        results.append(preds)

    return results

def process_line(batch):
    # return [(text_e[0], predict_top3(text_e[1], text_e[0])) for text_e in batch]
    lines = [x[1] for x in batch]
    idxs = [x[0] for x in batch]
    return zip(idxs, predict_top3_batch(lines))

def batch_list(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def run_parallel_inference(lines, work_dir, num_workers=1):
    # with mp.Pool(
    #     processes=num_workers,
    #     initializer=init_worker,
    #     initargs=(work_dir,)
    # ) as pool:
    #     lines_e = list(enumerate(lines))
    #     batches = list(batch_list(lines_e, 256))
    #     results = pool.map(process_line, batches)

    init_worker(work_dir)
    results = []
    lines_e = list(enumerate(lines))
    lines_e.sort(key=lambda x: len(x[1]))
    batches = batch_list(lines_e, 128)
    progress = 0
    for b in batches:
        results.extend(process_line(b))
        progress += len(b)
        print(f"PROGRESS: {progress}/{len(lines)}")
    return results

    res = []
    for k in results:
        res.extend(k)
    return res