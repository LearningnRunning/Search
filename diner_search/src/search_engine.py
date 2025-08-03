"""
음식점 검색 엔진
"""

import logging
from typing import Any, Optional

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from src.utils import jamo_similarity, normalize
from src.embedding_loader import EmbeddingLoader, load_embeddings

# 로거 설정
logger = logging.getLogger(__name__)


class SemanticSearcher:
    """의미론적 검색을 위한 클래스"""

    def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        """
        Args:
            model_name: 사용할 SBERT 모델명
        """
        try:
            self.model = SentenceTransformer(model_name)
            # 디바이스 정보 로깅
            device = next(self.model.parameters()).device
            logger.info(f"SBERT 모델이 {device}에서 로드되었습니다.")
        except Exception as e:
            logger.error(f"SBERT 모델 로드 실패: {e}")
            logger.info("로컬 모델 또는 오프라인 모드로 전환합니다.")
            # 간단한 로컬 모델 또는 None으로 설정
            self.model = None

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
        if self.model is None:
            # 모델이 없을 때는 간단한 문자열 매칭으로 대체
            logger.warning("SBERT 모델이 없어 간단한 문자열 매칭을 사용합니다.")
            results = []
            for candidate in candidates:
                # 간단한 포함 관계 기반 점수 계산
                if query.lower() in candidate.lower():
                    score = len(query) / len(candidate)
                    results.append((candidate, score))
            return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        
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
        
        # semantic_searcher는 항상 초기화 (의미론적 검색에 필요)
        logger.info("🤖 SBERT 모델 로드 중...")
        self.semantic_searcher = SemanticSearcher(self.model_name)
        
        # 미리 계산된 벡터 사용 시도
        if use_precomputed_embeddings:
            self._load_precomputed_embeddings()
        
        # 벡터 로드 실패 시 기존 방식 사용
        if self.diner_embeddings is None:
            self._initialize_with_model()
        
        # 디바이스 통일 확인
        self._ensure_device_consistency()
    
    def _ensure_device_consistency(self):
        """모든 텐서가 같은 디바이스에 있는지 확인하고 통일합니다."""
        if self.diner_embeddings is not None:
            model_device = next(self.semantic_searcher.model.parameters()).device
            if self.diner_embeddings.device != model_device:
                logger.info(f"텐서 디바이스 통일: {self.diner_embeddings.device} -> {model_device}")
                self.diner_embeddings = self.diner_embeddings.to(model_device)
    
    def _load_precomputed_embeddings(self):
        """미리 계산된 벡터를 로드합니다."""
        try:
            logger.info("🔄 미리 계산된 벡터 로드 중...")
            self.embedding_loader = load_embeddings(self.embeddings_dir)
            
            if self.embedding_loader and self.embedding_loader.is_loaded():
                self.diner_embeddings = self.embedding_loader.get_embeddings()
                if self.diner_infos is None:
                    self.diner_infos = self.embedding_loader.get_diner_infos()
                
                info = self.embedding_loader.get_info()
                logger.info(f"✅ 미리 계산된 벡터 로드 완료!")
                logger.info(f"   음식점 수: {info['num_diners']:,}개")
                logger.info(f"   벡터 차원: {info['embedding_dim']}")
                logger.info(f"   파일 크기: {info['file_size_mb']:.1f}MB")
                logger.info(f"   모델: {info['model_name']}")
            else:
                logger.warning("⚠️ 미리 계산된 벡터를 찾을 수 없습니다. 모델을 사용하여 초기화합니다.")
                
        except Exception as e:
            logger.error(f"❌ 벡터 로드 중 오류 발생: {e}")
            logger.info("🔄 모델을 사용하여 초기화합니다.")
    
    def _initialize_with_model(self):
        """모델을 사용하여 벡터를 초기화합니다."""
        logger.info("🤖 SBERT 모델을 사용하여 벡터 초기화 중...")
        
        if self.diner_infos is None:
            from src.utils import load_diner_data
            self.diner_infos = load_diner_data()
        
        # 모델이 없을 때는 벡터 초기화를 건너뜀
        if self.semantic_searcher.model is None:
            logger.warning("⚠️ SBERT 모델이 없어 벡터 초기화를 건너뜁니다.")
            self.diner_embeddings = None
            return
        
        # 음식점 이름들의 임베딩을 미리 계산
        diner_names = [d["name"] for d in self.diner_infos]
        logger.info(f"🔢 {len(diner_names)}개 음식점의 벡터 생성 중...")
        self.diner_embeddings = self.semantic_searcher.model.encode(
            diner_names, convert_to_tensor=True
        )
        logger.info("✅ 벡터 생성 완료!")

    def search(
        self, query: str, top_k: int = 5, jamo_threshold: float = 0.9
    ) -> pd.DataFrame:
        """
        음식점을 검색합니다.

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 수

        Returns:
            검색 결과 DataFrame
        """
        norm_query = normalize(query)

        # 1. 정확한 매칭
        exact_matches = [
            d for d in self.diner_infos if normalize(d["name"]) == norm_query
        ]
        if exact_matches:
            results = pd.DataFrame(exact_matches).assign(match_type="정확한 매칭")
            return self._add_kakao_map_links(results)

        # 2. 부분 매칭
        partial_matches = [
            d for d in self.diner_infos if norm_query in normalize(d["name"])
        ]
        if partial_matches:
            results = pd.DataFrame(partial_matches).assign(match_type="부분 매칭")
            return self._add_kakao_map_links(results)

        # 3. 자모 기반 직접 매칭
        for d in self.diner_infos:
            is_jamo, score = jamo_similarity(norm_query, normalize(d["name"]))
            if is_jamo:
                results = pd.DataFrame(
                    [
                        {
                            "name": d["name"],
                            "idx": d["idx"],
                            "jamo_score": score,
                            "match_type": "자모 매칭",
                        }
                    ]
                )
                return self._add_kakao_map_links(results)

        # 4. 자모 유사도 전체 계산 (기본 임계값 0.7 사용)
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
        sbert_top = []
        if self.semantic_searcher.model is not None and self.diner_embeddings is not None:
            try:
                # 쿼리 임베딩 생성
                query_emb = self.semantic_searcher.model.encode(query, convert_to_tensor=True)
                
                # 디바이스 통일: diner_embeddings를 query_emb와 같은 디바이스로 이동
                if self.diner_embeddings.device != query_emb.device:
                    logger.info(f"검색 중 텐서 디바이스 통일: {self.diner_embeddings.device} -> {query_emb.device}")
                    self.diner_embeddings = self.diner_embeddings.to(query_emb.device)
                
                cos_scores = util.cos_sim(query_emb, self.diner_embeddings)[0]
                topk = torch.topk(cos_scores, k=top_k)
                sbert_top = [
                    (self.diner_infos[idx], round(score.item(), 4))
                    for idx, score in zip(topk.indices.cpu(), topk.values.cpu(), strict=False)
                ]
                
            except Exception as e:
                logger.error(f"의미론적 검색 중 오류 발생: {e}")
                sbert_top = []
        else:
            logger.info("SBERT 모델이 없어 의미론적 검색을 건너뜁니다.")
        
        # 의미론적 검색 실패 시 자모 검색 결과만 반환
        if not sbert_top and jamo_top:
            results = pd.DataFrame([
                {
                    "name": d["name"],
                    "idx": d["idx"],
                    "score": score,
                    "match_type": "자모 매칭",
                }
                for d, score in jamo_top
            ])
            return self._add_kakao_map_links(results)
        elif not sbert_top and not jamo_top:
            return pd.DataFrame().assign(match_type="검색 결과 없음")

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

        results = pd.DataFrame(
            sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        ).assign(match_type="통합 검색")
        
        return self._add_kakao_map_links(results)
    
    def _add_kakao_map_links(self, df: pd.DataFrame) -> pd.DataFrame:
        """검색 결과에 카카오맵 링크를 추가합니다."""
        if df.empty:
            return df
        
        # 카카오맵 링크 컬럼 추가
        df = df.copy()
        # name 컬럼을 카카오맵 링크로 변환
        df['name'] = df.apply(lambda row: f"[{row['name']}](https://place.map.kakao.com/{row['idx']})", axis=1)
        
        return df
    
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
