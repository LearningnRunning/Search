"""
저장된 음식점 벡터를 로드하는 클래스
"""

import os
import json
import pickle
import torch
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class EmbeddingLoader:
    """저장된 음식점 벡터를 로드하는 클래스"""
    
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        """
        Args:
            embeddings_dir: 벡터 파일들이 저장된 디렉토리
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings = None
        self.metadata = None
        self.diner_names = None
        self.diner_infos = None
        
    def load_pt_format(self) -> bool:
        """
        .pt 형식의 벡터 파일을 로드합니다.
        
        Returns:
            로드 성공 여부
        """
        pt_path = self.embeddings_dir / "diner_embeddings.pt"
        metadata_path = self.embeddings_dir / "diner_metadata.json"
        
        if not pt_path.exists():
            print(f"❌ 벡터 파일을 찾을 수 없습니다: {pt_path}")
            return False
            
        if not metadata_path.exists():
            print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
            return False
        
        try:
            # 벡터 로드
            self.embeddings = torch.load(pt_path)
            print(f"✅ 벡터 로드 완료: {self.embeddings.shape}")
            
            # 메타데이터 로드
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            self.diner_names = self.metadata["diner_names"]
            self.diner_infos = self.metadata["diner_infos"]
            
            print(f"✅ 메타데이터 로드 완료: {len(self.diner_names)}개 음식점")
            return True
            
        except Exception as e:
            print(f"❌ 파일 로드 중 오류 발생: {e}")
            return False
    
    def load_pkl_format(self) -> bool:
        """
        .pkl 형식의 벡터 파일을 로드합니다.
        
        Returns:
            로드 성공 여부
        """
        pkl_path = self.embeddings_dir / "diner_embeddings.pkl"
        
        if not pkl_path.exists():
            print(f"❌ 벡터 파일을 찾을 수 없습니다: {pkl_path}")
            return False
        
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            self.embeddings = data["embeddings"]
            self.metadata = data["metadata"]
            self.diner_names = self.metadata["diner_names"]
            self.diner_infos = self.metadata["diner_infos"]
            
            print(f"✅ 벡터 로드 완료: {self.embeddings.shape}")
            print(f"✅ 메타데이터 로드 완료: {len(self.diner_names)}개 음식점")
            return True
            
        except Exception as e:
            print(f"❌ 파일 로드 중 오류 발생: {e}")
            return False
    
    def load(self, prefer_pkl: bool = True) -> bool:
        """
        벡터 파일을 로드합니다. 기본적으로 .pkl 형식을 선호합니다.
        
        Args:
            prefer_pkl: .pkl 형식을 우선적으로 로드할지 여부
            
        Returns:
            로드 성공 여부
        """
        if prefer_pkl:
            # .pkl 형식 먼저 시도
            if self.load_pkl_format():
                return True
            # 실패하면 .pt 형식 시도
            return self.load_pt_format()
        else:
            # .pt 형식 먼저 시도
            if self.load_pt_format():
                return True
            # 실패하면 .pkl 형식 시도
            return self.load_pkl_format()
    
    def get_embeddings(self) -> Optional[torch.Tensor]:
        """저장된 벡터를 반환합니다."""
        return self.embeddings
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """메타데이터를 반환합니다."""
        return self.metadata
    
    def get_diner_names(self) -> Optional[list]:
        """음식점 이름 리스트를 반환합니다."""
        return self.diner_names
    
    def get_diner_infos(self) -> Optional[list]:
        """음식점 정보 리스트를 반환합니다."""
        return self.diner_infos
    
    def get_diner_name_by_index(self, index: int) -> Optional[str]:
        """인덱스로 음식점 이름을 가져옵니다."""
        if self.diner_names and 0 <= index < len(self.diner_names):
            return self.diner_names[index]
        return None
    
    def get_diner_info_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """인덱스로 음식점 정보를 가져옵니다."""
        if self.diner_infos and 0 <= index < len(self.diner_infos):
            return self.diner_infos[index]
        return None
    
    def get_embedding_by_index(self, index: int) -> Optional[torch.Tensor]:
        """인덱스로 벡터를 가져옵니다."""
        if self.embeddings is not None and 0 <= index < len(self.embeddings):
            return self.embeddings[index]
        return None
    
    def get_embedding_by_name(self, name: str) -> Optional[Tuple[int, torch.Tensor]]:
        """이름으로 벡터를 가져옵니다."""
        if self.diner_names is None:
            return None
        
        try:
            index = self.diner_names.index(name)
            return index, self.embeddings[index]
        except ValueError:
            return None
    
    def is_loaded(self) -> bool:
        """벡터가 로드되었는지 확인합니다."""
        return self.embeddings is not None and self.metadata is not None
    
    def get_info(self) -> Dict[str, Any]:
        """로드된 벡터의 정보를 반환합니다."""
        if not self.is_loaded():
            return {"loaded": False}
        
        return {
            "loaded": True,
            "num_diners": len(self.diner_names),
            "embedding_dim": self.embeddings.shape[1],
            "model_name": self.metadata.get("model_name", "Unknown"),
            "file_size_mb": self.embeddings.element_size() * self.embeddings.numel() / (1024 * 1024)
        }


def load_embeddings(embeddings_dir: str = "data/embeddings", prefer_pkl: bool = True) -> Optional[EmbeddingLoader]:
    """
    벡터를 로드하는 편의 함수
    
    Args:
        embeddings_dir: 벡터 파일들이 저장된 디렉토리
        prefer_pkl: .pkl 형식을 우선적으로 로드할지 여부
        
    Returns:
        로드된 EmbeddingLoader 인스턴스 또는 None
    """
    loader = EmbeddingLoader(embeddings_dir)
    if loader.load(prefer_pkl):
        return loader
    return None 