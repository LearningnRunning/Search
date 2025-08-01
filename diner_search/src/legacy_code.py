# advanced_diner_search.py

import re

import pandas as pd
import torch
from jamo import hangul_to_jamo
from sentence_transformers import SentenceTransformer, util


def normalize(text):
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", text.lower().strip())

def jamo_similarity(a: str, b: str) -> float:
    a_jamo = ' '.join(hangul_to_jamo(a))
    b_jamo = ' '.join(hangul_to_jamo(b))
    score = fuzz.ratio(a_jamo, b_jamo)
    if score > 80:
        return True, score
    else:
        matches = sum(x == y for x, y in zip(a_jamo, b_jamo))
        return False, matches / max(len(a_jamo), len(b_jamo), 1)

class SemanticSearcher:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        self.model = SentenceTransformer(model_name)

    def similarity(self, query: str, candidates: list[str], top_k=5) -> list[tuple[str, float]]:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return [(candidates[idx], round(score.item(), 4)) for idx, score in zip(top_results.indices, top_results.values)]

def advanced_diner_search_cached(query, diner_infos, diner_embs, top_k=5, jamo_threshold=0.7):
    norm_query = normalize(query)
    diner_names = [d["name"] for d in diner_infos]

    # 1. Exact match
    exact_matches = [d for d in diner_infos if normalize(d["name"]) == norm_query]
    if exact_matches:
        return pd.DataFrame(exact_matches).assign(match_type="exact")

    # 2. Partial match
    partial_matches = [d for d in diner_infos if norm_query in normalize(d["name"])]
    if partial_matches:
        return pd.DataFrame(partial_matches).assign(match_type="partial")

    # 3. Jamo-based direct match
    for d in diner_infos:
        is_jamo, score = jamo_similarity(norm_query, normalize(d["name"]))
        if is_jamo:
            return pd.DataFrame([{
                "diner_name": d["name"],
                "diner_idx": d["idx"],
                "jamo_score": score,
                "match_type": "jamo"
            }])

    # 4. Jamo 유사도 전체 계산
    jamo_scores = []
    for d in diner_infos:
        _, score = jamo_similarity(norm_query, normalize(d["name"]))
        jamo_scores.append((d, score))

    jamo_top = sorted([x for x in jamo_scores if x[1] >= jamo_threshold], key=lambda x: x[1], reverse=True)[:top_k]

    # 5. SBERT
    query_emb = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, diner_embs)[0]
    topk = torch.topk(cos_scores, k=top_k)
    sbert_top = [(diner_infos[idx], round(score.item(), 4))
                 for idx, score in zip(topk.indices.cpu(), topk.values.cpu())]

    # 6. 점수 합산
    combined = {}
    for d, score in jamo_top:
        key = d["name"]
        combined[key] = {"diner_name": key, "diner_idx": d["idx"], "score": score * 0.5}

    for d, score in sbert_top:
        key = d["name"]
        if key in combined:
            combined[key]["score"] += score * 0.5
        else:
            combined[key] = {"diner_name": key, "diner_idx": d["idx"], "score": score * 0.5}

    return pd.DataFrame(
        sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    ).assign(match_type="combined")

import json

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# DataFrame에서 diner_infos 생성
with open("diner_infos.json", "r", encoding="utf-8") as f:
    loaded_diner_infos = json.load(f)
    
diner_embs = model.encode([d["name"] for d in diner_infos], convert_to_tensor=True)

advanced_diner_search_cached('윤씨네', diner_infos, diner_embs)
