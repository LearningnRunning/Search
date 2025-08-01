# advanced_diner_search.py
# 참조용 레거시 코드 - 실제 사용하지 않음

import re

import pandas as pd
import torch
from jamo import hangul_to_jamo
from sentence_transformers import SentenceTransformer, util


def normalize(text):
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", text.lower().strip())


def jamo_similarity(a: str, b: str) -> tuple[bool, float]:
    a_jamo = " ".join(hangul_to_jamo(a))
    b_jamo = " ".join(hangul_to_jamo(b))
    # fuzz import가 누락되어 있어서 주석 처리
    # score = fuzz.ratio(a_jamo, b_jamo)
    score = 0  # 임시 값
    if score > 80:
        return True, score
    else:
        matches = sum(x == y for x, y in zip(a_jamo, b_jamo, strict=False))
        return False, matches / max(len(a_jamo), len(b_jamo), 1)


class SemanticSearcher:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        self.model = SentenceTransformer(model_name)

    def similarity(
        self, query: str, candidates: list[str], top_k=5
    ) -> list[tuple[str, float]]:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return [
            (candidates[idx], round(score.item(), 4))
            for idx, score in zip(top_results.indices, top_results.values, strict=False)
        ]


def advanced_diner_search_cached(
    query, diner_infos, diner_embs, top_k=5, jamo_threshold=0.7
):
    norm_query = normalize(query)

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
            return pd.DataFrame(
                [
                    {
                        "diner_name": d["name"],
                        "diner_idx": d["idx"],
                        "jamo_score": score,
                        "match_type": "jamo",
                    }
                ]
            )

    # 4. Jamo 유사도 전체 계산
    jamo_scores = []
    for d in diner_infos:
        _, score = jamo_similarity(norm_query, normalize(d["name"]))
        jamo_scores.append((d, score))

    jamo_top = sorted(
        [x for x in jamo_scores if x[1] >= jamo_threshold],
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # 5. SBERT (참조용 - 실제로는 사용하지 않음)
    # model 변수가 정의되지 않아서 주석 처리
    # query_emb = model.encode(query, convert_to_tensor=True)
    # cos_scores = util.cos_sim(query_emb, diner_embs)[0]
    # topk = torch.topk(cos_scores, k=top_k)
    # sbert_top = [
    #     (diner_infos[idx], round(score.item(), 4))
    #     for idx, score in zip(topk.indices.cpu(), topk.values.cpu(), strict=False)
    # ]

    # 6. 점수 합산 (참조용 - 실제로는 jamo_top만 사용)
    combined = {}
    for d, score in jamo_top:
        key = d["name"]
        combined[key] = {"diner_name": key, "diner_idx": d["idx"], "score": score * 0.5}

    return pd.DataFrame(
        sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    ).assign(match_type="combined")


# 참조용 코드 - 실제 실행하지 않음
if __name__ == "__main__":
    print("이 파일은 참조용입니다. 실제 사용하지 않습니다.")
