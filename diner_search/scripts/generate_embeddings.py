"""
음식점 이름 벡터 생성 및 저장 스크립트
"""

import json
import os
import sys
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils import load_diner_data


def generate_and_save_embeddings(
    model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    output_dir: str = "data/embeddings",
    batch_size: int = 32
):
    """
    음식점 이름의 벡터를 생성하고 저장합니다.
    
    Args:
        model_name: 사용할 SBERT 모델명
        output_dir: 벡터 저장 디렉토리
        batch_size: 배치 크기
    """
    print(f"🚀 음식점 이름 벡터 생성 시작...")
    print(f"📦 모델: {model_name}")
    print(f"📁 저장 위치: {output_dir}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드
    print("📊 음식점 데이터 로드 중...")
    diner_infos = load_diner_data()
    print(f"✅ {len(diner_infos)}개의 음식점 데이터 로드 완료")
    
    # 모델 로드
    print("🤖 SBERT 모델 로드 중...")
    model = SentenceTransformer(model_name)
    print("✅ 모델 로드 완료")
    
    # 음식점 이름 추출
    diner_names = [d["name"] for d in diner_infos]
    
    # 벡터 생성
    print("🔢 벡터 생성 중...")
    embeddings = []
    
    for i in tqdm(range(0, len(diner_names), batch_size), desc="벡터 생성"):
        batch_names = diner_names[i:i + batch_size]
        batch_embeddings = model.encode(batch_names, convert_to_tensor=True)
        embeddings.append(batch_embeddings.cpu())
    
    # 모든 배치를 하나로 합치기
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"✅ 벡터 생성 완료: {all_embeddings.shape}")
    
    # 메타데이터 준비
    metadata = {
        "model_name": model_name,
        "num_diners": len(diner_infos),
        "embedding_dim": all_embeddings.shape[1],
        "diner_names": diner_names,
        "diner_infos": diner_infos
    }
    
    # 파일 저장
    print("💾 파일 저장 중...")
    
    # PyTorch 텐서로 저장 (.pt)
    torch.save(all_embeddings, os.path.join(output_dir, "diner_embeddings.pt"))
    print("✅ diner_embeddings.pt 저장 완료")
    
    # 메타데이터를 JSON으로 저장
    with open(os.path.join(output_dir, "diner_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("✅ diner_metadata.json 저장 완료")
    
    # 압축된 형태로도 저장 (.pkl)
    import pickle
    compressed_data = {
        "embeddings": all_embeddings,
        "metadata": metadata
    }
    with open(os.path.join(output_dir, "diner_embeddings.pkl"), "wb") as f:
        pickle.dump(compressed_data, f)
    print("✅ diner_embeddings.pkl 저장 완료")
    
    print(f"🎉 모든 파일 저장 완료!")
    print(f"📊 벡터 크기: {all_embeddings.shape}")
    print(f"💾 저장된 파일:")
    print(f"   - {output_dir}/diner_embeddings.pt")
    print(f"   - {output_dir}/diner_metadata.json")
    print(f"   - {output_dir}/diner_embeddings.pkl")


def verify_embeddings(embeddings_dir: str = "data/embeddings"):
    """
    생성된 벡터 파일들을 검증합니다.
    
    Args:
        embeddings_dir: 벡터 파일들이 저장된 디렉토리
    """
    print("🔍 벡터 파일 검증 중...")
    
    # .pt 파일 검증
    pt_path = os.path.join(embeddings_dir, "diner_embeddings.pt")
    if os.path.exists(pt_path):
        embeddings = torch.load(pt_path)
        print(f"✅ diner_embeddings.pt: {embeddings.shape}")
    else:
        print("❌ diner_embeddings.pt 파일이 없습니다.")
    
    # .pkl 파일 검증
    pkl_path = os.path.join(embeddings_dir, "diner_embeddings.pkl")
    if os.path.exists(pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ diner_embeddings.pkl: {data['embeddings'].shape}")
        print(f"   메타데이터: {len(data['metadata']['diner_names'])}개 음식점")
    else:
        print("❌ diner_embeddings.pkl 파일이 없습니다.")
    
    # JSON 메타데이터 검증
    json_path = os.path.join(embeddings_dir, "diner_metadata.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"✅ diner_metadata.json: {metadata['num_diners']}개 음식점")
        print(f"   모델: {metadata['model_name']}")
        print(f"   벡터 차원: {metadata['embedding_dim']}")
    else:
        print("❌ diner_metadata.json 파일이 없습니다.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="음식점 이름 벡터 생성")
    parser.add_argument("--model", default="snunlp/KR-SBERT-V40K-klueNLI-augSTS", 
                       help="사용할 SBERT 모델명")
    parser.add_argument("--output-dir", default="data/embeddings", 
                       help="벡터 저장 디렉토리")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="배치 크기")
    parser.add_argument("--verify", action="store_true", 
                       help="기존 벡터 파일 검증")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_embeddings(args.output_dir)
    else:
        generate_and_save_embeddings(
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        ) 