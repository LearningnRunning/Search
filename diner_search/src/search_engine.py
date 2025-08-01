"""
음식점 검색 엔진
"""

from typing import Any

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from utils import jamo_similarity, normalize


class SemanticSearcher:
    """의미론적 검색을 위한 클래스"""

    def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        """
        Args:
            model_name: 사용할 SBERT 모델명
        """
        self.model = SentenceTransformer(model_name)

    def similarity(
        self, query: str, candidates: list[str], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        쿼리와 후보들 간의 의미론적 유사도를 계산합니다.

        Args:
            query: 검색 쿼리
            candidates: 후보 문자열들
            top_k: 반환할 상위 결과 수

        Returns:
            (후보, 점수) 튜플의 리스트
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return [
            (candidates[idx], round(score.item(), 4))
            for idx, score in zip(top_results.indices, top_results.values, strict=False)
        ]


class DinerSearchEngine:
    """음식점 검색 엔진"""

    def __init__(
        self,
        diner_infos: list[dict[str, Any]],
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    ):
        """
        Args:
            diner_infos: 음식점 정보 리스트
            model_name: SBERT 모델명
        """
        self.diner_infos = diner_infos
        self.semantic_searcher = SemanticSearcher(model_name)

        # 음식점 이름들의 임베딩을 미리 계산
        diner_names = [d["name"] for d in diner_infos]
        self.diner_embeddings = self.semantic_searcher.model.encode(
            diner_names, convert_to_tensor=True
        )

    def search(
        self, query: str, top_k: int = 5, jamo_threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        음식점을 검색합니다.

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 수
            jamo_threshold: 자모 유사도 임계값

        Returns:
            검색 결과 DataFrame
        """
        norm_query = normalize(query)
        diner_names = [d["name"] for d in self.diner_infos]

        # 1. 정확한 매칭
        exact_matches = [
            d for d in self.diner_infos if normalize(d["name"]) == norm_query
        ]
        if exact_matches:
            return pd.DataFrame(exact_matches).assign(match_type="정확한 매칭")

        # 2. 부분 매칭
        partial_matches = [
            d for d in self.diner_infos if norm_query in normalize(d["name"])
        ]
        if partial_matches:
            return pd.DataFrame(partial_matches).assign(match_type="부분 매칭")

        # 3. 자모 기반 직접 매칭
        for d in self.diner_infos:
            is_jamo, score = jamo_similarity(norm_query, normalize(d["name"]))
            if is_jamo:
                return pd.DataFrame(
                    [
                        {
                            "name": d["name"],
                            "idx": d["idx"],
                            "jamo_score": score,
                            "match_type": "자모 매칭",
                        }
                    ]
                )

        # 4. 자모 유사도 전체 계산
        jamo_scores = []
        for d in self.diner_infos:
            _, score = jamo_similarity(norm_query, normalize(d["name"]))
            jamo_scores.append((d, score))

        jamo_top = sorted(
            [x for x in jamo_scores if x[1] >= jamo_threshold],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        # 5. SBERT 의미론적 검색
        query_emb = self.semantic_searcher.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.diner_embeddings)[0]
        topk = torch.topk(cos_scores, k=top_k)
        sbert_top = [
            (self.diner_infos[idx], round(score.item(), 4))
            for idx, score in zip(topk.indices.cpu(), topk.values.cpu(), strict=False)
        ]

        # 6. 점수 합산 (통합 검색)
        combined = {}
        for d, score in jamo_top:
            key = d["name"]
            combined[key] = {"name": key, "idx": d["idx"], "score": score * 0.5}

        for d, score in sbert_top:
            key = d["name"]
            if key in combined:
                combined[key]["score"] += score * 0.5
            else:
                combined[key] = {"name": key, "idx": d["idx"], "score": score * 0.5}

        return pd.DataFrame(
            sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        ).assign(match_type="통합 검색")
