from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from scipy.spatial.distance import cosine
import torch
import numpy as np


class CosSimilarity:
    """
    用于衡量文本质量的余弦相似度计算类。
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.detach().cpu().numpy()
    
    def similarity(self, text1: str, text2: str) -> float:
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        sim = 1 - cosine(emb1, emb2)
        return sim
    
    
    def __call__(self, text1: str, text2: str) -> float:
        return self.similarity(text1, text2)

if __name__ == "__main__":
    cos_sim = CosSimilarity()
    baseline_file = "exp/vicuna/large_mt_bench.jsonl"
    import json
    with open(baseline_file, "r") as f:
        baseline_lines = f.readlines()
        baseline_data = [json.loads(line) for line in baseline_lines]
    tridec_file = "exp/vicuna/sd_mt_bench.jsonl"
    with open(tridec_file, "r") as f:
        tridec_lines = f.readlines()
        tridec_data = [json.loads(line) for line in tridec_lines]

    scores = []
    for base, tri in zip(baseline_data, tridec_data):
        base_data: str = base["choices"][0]["turns"][0]
        tri_data: str = tri["choices"][0]["turns"][0]
        score = cos_sim(base_data, tri_data)
        scores.append(score)
    print(f"Average Cosine Similarity: {np.mean(scores):.4f}")
