"""
음식점 검색 시스템 - Gradio 애플리케이션
로컬 개발 및 Hugging Face Spaces 배포용
"""

import gradio as gr
import pandas as pd
import os
import sys
import logging

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# src 디렉토리를 Python 경로에 추가 (Hugging Face Spaces 배포용)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if os.path.exists(src_path):
    sys.path.append(src_path)

from src.search_engine import DinerSearchEngine
from src.utils import load_diner_data


def load_search_engine():
    """검색 엔진을 로드합니다."""
    try:
        # 미리 계산된 벡터를 사용하여 검색 엔진 초기화
        search_engine = DinerSearchEngine(
            diner_infos=None,  # 벡터 파일에서 자동 로드
            use_precomputed_embeddings=True,  # 미리 계산된 벡터 사용
            embeddings_dir="data/embeddings"
        )
        
        # 검색 엔진 정보 출력
        info = search_engine.get_info()
        logger.info(f"🔍 검색 엔진 초기화 완료:")
        logger.info(f"   모델: {info.get('model_name', 'Unknown')}")
        logger.info(f"   음식점 수: {info.get('num_diners', 0):,}개")
        logger.info(f"   미리 계산된 벡터 사용: {info.get('use_precomputed_embeddings', False)}")
        
        if info.get('loaded', False):
            logger.info(f"   벡터 차원: {info.get('embedding_dim', 0)}")
            logger.info(f"   파일 크기: {info.get('file_size_mb', 0):.1f}MB")
        
        return search_engine
    except Exception as e:
        logger.error(f"❌ 검색 엔진 로드 실패: {str(e)}")
        return None


def get_dataframe_datatypes(df):
    """DataFrame의 컬럼별 datatype을 반환합니다."""
    datatypes = []
    for col in df.columns:
        if col == "카카오맵":
            datatypes.append("markdown")
        else:
            datatypes.append("str")
    return datatypes

def search_diners(query, top_k, search_method, accuracy_level):
    """
    음식점을 검색합니다.
    
    Args:
        query: 검색 쿼리
        top_k: 검색 결과 수
        search_method: 검색 방법
        accuracy_level: 검색 정확도
        
    Returns:
        검색 결과 DataFrame과 링크 HTML
    """
    if not query.strip():
        return "검색어를 입력해주세요.", pd.DataFrame(), ""
    
    try:
        # 검색 실행
        results = search_engine.search(
            query=query.strip(), 
            top_k=top_k,
            jamo_threshold=0.9  # 기본값으로 설정
        )
        
        logger.info(f"검색 결과 컬럼: {list(results.columns)}")
        
        if not results.empty:
            # 점수 컬럼 포맷팅
            display_df = results.copy()
            if "score" in display_df.columns:
                display_df["score"] = display_df["score"].apply(lambda x: f"{x:.4f}")
            if "jamo_score" in display_df.columns:
                display_df["jamo_score"] = display_df["jamo_score"].apply(lambda x: f"{x:.2f}")
           
            # 매칭 타입별 통계
            match_stats = results["match_type"].value_counts()
            stats_text = "📊 매칭 타입별 통계:\n"
            for match_type, count in match_stats.items():
                stats_text += f"- {match_type}: {count}개\n"
            
            # name 컬럼이 이미 카카오맵 링크로 변환되어 있으므로 별도 링크 영역은 제거
            links_html = "<div style='color: #666; font-style: italic;'>💡 음식점 이름을 클릭하면 카카오맵으로 이동합니다.</div>"
            
            return f"✅ '{query}'에 대한 검색 결과 ({len(results)}개)", display_df, links_html
        else:
            return f"⚠️ '{query}'에 대한 검색 결과가 없습니다.", pd.DataFrame(), "<div style='color: #666; font-style: italic;'>검색 결과가 없어 카카오맵 링크를 제공할 수 없습니다.</div>"
            
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {str(e)}")
        return f"❌ 검색 중 오류가 발생했습니다: {str(e)}", pd.DataFrame(), "<div style='color: #666; font-style: italic;'>오류로 인해 카카오맵 링크를 제공할 수 없습니다.</div>"


# 검색 엔진 로드
search_engine = load_search_engine()

# Gradio 인터페이스 구성
def create_interface():
    """Gradio 인터페이스를 생성합니다."""
    
    with gr.Blocks(
        title="🍽️ 음식점 검색 시스템",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .dataframe-container {
            max-height: 500px;
            overflow-y: auto;
        }
        .kakao-map-link {
            color: #FEE500 !important;
            font-weight: bold !important;
            text-decoration: none !important;
            background-color: #000 !important;
            padding: 4px 8px !important;
            border-radius: 4px !important;
            display: inline-block !important;
        }
        .kakao-map-link:hover {
            background-color: #FEE500 !important;
            color: #000 !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🍽️ 음식점 검색 시스템
        
        다양한 검색 방법으로 원하는 음식점을 찾아보세요!
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # 검색 입력
                query_input = gr.Textbox(
                    label="음식점 이름을 입력하세요",
                    placeholder="예: 윤씨네, 맛있는집, 피자헛...",
                    lines=1
                )
                
                # 검색 버튼
                search_btn = gr.Button("🔍 검색", variant="primary", size="lg")
            
            # with gr.Column(scale=1):
                # # 검색 설정
                # with gr.Group():
                #     gr.Markdown("### 🔧 검색 설정")
                    
                #     top_k_slider = gr.Slider(
                #         minimum=1,
                #         maximum=20,
                #         value=5,
                #         step=1,
                #         label="검색 결과 수",
                #         info="표시할 검색 결과의 개수"
                #     )
                    
                #     search_method_dropdown = gr.Dropdown(
                #         choices=["자동 선택", "정확한 매칭", "부분 매칭", "자모 매칭", "의미론적 검색", "통합 검색"],
                #         value="자동 선택",
                #         label="검색 방법",
                #         info="사용할 검색 방법을 선택하세요"
                #     )
                    
                #     accuracy_level_slider = gr.Slider(
                #         minimum=1,
                #         maximum=5,
                #         value=3,
                #         step=1,
                #         label="검색 정확도",
                #         info="1: 빠른 검색, 5: 정확한 검색"
                #     )
        
        # 결과 출력
        with gr.Row():
            with gr.Column():
                result_text = gr.Textbox(
                    label="검색 결과",
                    interactive=False,
                    lines=2
                )
                
                # 카카오맵 링크 영역
                links_html = gr.HTML(
                    label="🗺️ 카카오맵 링크",
                    visible=True
                )
                
                result_table = gr.Dataframe(
                    label="📋 검색 결과",
                    interactive=False,
                    wrap=True,
                    line_breaks=False,
                    datatype=["markdown", "str", "str", "str"]  # name(링크), idx, score, match_type
                )
        
        # 검색 방법 설명
        with gr.Accordion("💡 검색 방법 설명", open=False):
            gr.Markdown("""
            ### 검색 방법:
            
            - **정확한 매칭**: 완전히 일치하는 음식점
            - **부분 매칭**: 검색어가 포함된 음식점  
            - **자모 매칭**: 한글 자모 유사도 기반
            - **의미론적 검색**: SBERT 모델을 사용한 의미 기반 검색
            - **통합 검색**: 자모 + 의미론적 검색 조합
            - **자동 선택**: 시스템이 자동으로 최적의 방법 선택
            
            ### 검색 예시:
            - **정확한 매칭**: "윤씨네" → 정확히 "윤씨네"라는 이름의 음식점
            - **부분 매칭**: "피자" → "피자헛", "도미노피자" 등 "피자"가 포함된 음식점
            - **자모 매칭**: "윤씨네" → "윤시네", "윤씨내" 등 비슷한 발음의 음식점
            - **의미론적 검색**: "맛있는 집" → "맛집", "좋은 음식점" 등 의미적으로 유사한 음식점
            
            ### 🗺️ 카카오맵 링크:
            검색 결과에서 "지도 보기" 버튼을 클릭하면 카카오맵에서 해당 음식점의 위치를 확인할 수 있습니다.
            """)
        
        # 이벤트 연결
        search_btn.click(
            fn=search_diners,
            inputs=[query_input],
            outputs=[result_text, result_table, links_html]
        )
        
        # Enter 키로도 검색 가능하도록
        query_input.submit(
            fn=search_diners,
            inputs=[query_input],
            outputs=[result_text, result_table, links_html]
        )
        
        # 푸터
        gr.Markdown("""
        ---
        <div style='text-align: center; color: #666;'>
            🍽️ 음식점 검색 시스템 | 다양한 검색 방법으로 원하는 음식점을 찾아보세요!
        </div>
        """)
    
    return demo


# Gradio 앱 실행
if __name__ == "__main__":
    demo = create_interface()
    
    # 환경에 따른 실행 설정
    if os.getenv("HF_SPACE_ID"):
        # Hugging Face Spaces에서 실행
        demo.launch()
    else:
        # 로컬 개발 환경에서 실행
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        ) 