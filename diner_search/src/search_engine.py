"""
음식점 검색 엔진
"""

from typing import Any, Optional

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from .utils import jamo_similarity, normalize
from .embedding_loader import EmbeddingLoader, load_embeddings


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
        diner_infos: Optional[list[dict[str, Any]]] = None,
        model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        embeddings_dir: str = "data/embeddings",
        use_precomputed_embeddings: bool = True
    ):
        """
        Args:
            diner_infos: 음식점 정보 리스트 (None이면 벡터 파일에서 로드)
            model_name: SBERT 모델명
            embeddings_dir: 벡터 파일들이 저장된 디렉토리
            use_precomputed_embeddings: 미리 계산된 벡터를 사용할지 여부
        """
        self.model_name = model_name
        self.embeddings_dir = embeddings_dir
        self.use_precomputed_embeddings = use_precomputed_embeddings
        
        # 벡터 로더 초기화
        self.embedding_loader = None
        self.diner_embeddings = None
        
        if diner_infos is not None:
            self.diner_infos = diner_infos
        else:
            self.diner_infos = None
        
        # 미리 계산된 벡터 사용 시도
        if use_precomputed_embeddings:
            self._load_precomputed_embeddings()
        
        # 벡터 로드 실패 시 기존 방식 사용
        if self.diner_embeddings is None:
            self._initialize_with_model()
    
    def _load_precomputed_embeddings(self):
        """미리 계산된 벡터를 로드합니다."""
        try:
            print("🔄 미리 계산된 벡터 로드 중...")
            self.embedding_loader = load_embeddings(self.embeddings_dir)
            
            if self.embedding_loader and self.embedding_loader.is_loaded():
                self.diner_embeddings = self.embedding_loader.get_embeddings()
                if self.diner_infos is None:
                    self.diner_infos = self.embedding_loader.get_diner_infos()
                
                info = self.embedding_loader.get_info()
                print(f"✅ 미리 계산된 벡터 로드 완료!")
                print(f"   음식점 수: {info['num_diners']:,}개")
                print(f"   벡터 차원: {info['embedding_dim']}")
                print(f"   파일 크기: {info['file_size_mb']:.1f}MB")
                print(f"   모델: {info['model_name']}")
            else:
                print("⚠️ 미리 계산된 벡터를 찾을 수 없습니다. 모델을 사용하여 초기화합니다.")
                
        except Exception as e:
            print(f"❌ 벡터 로드 중 오류 발생: {e}")
            print("🔄 모델을 사용하여 초기화합니다.")
    
    def _initialize_with_model(self):
        """모델을 사용하여 벡터를 초기화합니다."""
        print("🤖 SBERT 모델 로드 중...")
        self.semantic_searcher = SemanticSearcher(self.model_name)
        
        if self.diner_infos is None:
            from .utils import load_diner_data
            self.diner_infos = load_diner_data()
        
        # 음식점 이름들의 임베딩을 미리 계산
        diner_names = [d["name"] for d in self.diner_infos]
        print(f"🔢 {len(diner_names)}개 음식점의 벡터 생성 중...")
        self.diner_embeddings = self.semantic_searcher.model.encode(
            diner_names, convert_to_tensor=True
        )
        print("✅ 벡터 생성 완료!")

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
        if self.embedding_loader:
            # 미리 계산된 벡터 사용
            query_emb = self.semantic_searcher.model.encode(query, convert_to_tensor=True)
        else:
            # 기존 방식
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
    
    def get_info(self) -> dict:
        """검색 엔진 정보를 반환합니다."""
        info = {
            "model_name": self.model_name,
            "num_diners": len(self.diner_infos),
            "use_precomputed_embeddings": self.use_precomputed_embeddings
        }
        
        if self.embedding_loader:
            info.update(self.embedding_loader.get_info())
        
        return info
