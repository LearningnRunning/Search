"""
음식점 검색 시스템 - Streamlit 애플리케이션
"""

import streamlit as st

from search_engine import DinerSearchEngine
from utils import load_diner_data


@st.cache_resource
def load_search_engine():
    """검색 엔진을 로드하고 캐시합니다."""
    with st.spinner("음식점 데이터를 로드하고 있습니다..."):
        diner_infos = load_diner_data()
        search_engine = DinerSearchEngine(diner_infos)
    return search_engine


def main():
    """메인 애플리케이션"""

    # 페이지 설정
    st.set_page_config(
        page_title="음식점 검색 시스템",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 헤더
    st.title("🍽️ 음식점 검색 시스템")
    st.markdown("---")

    # 사이드바 설정
    with st.sidebar:
        st.header("🔧 검색 설정")

        # 검색 결과 수 설정
        top_k = st.slider(
            "검색 결과 수",
            min_value=1,
            max_value=20,
            value=5,
            help="표시할 검색 결과의 개수",
        )

        # 자모 유사도 임계값 설정
        jamo_threshold = st.slider(
            "자모 유사도 임계값",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="자모 기반 검색에서 사용할 유사도 임계값",
        )

        st.markdown("---")
        st.markdown("### 📊 검색 방법")
        st.markdown("""
        - **정확한 매칭**: 완전히 일치하는 음식점
        - **부분 매칭**: 검색어가 포함된 음식점
        - **자모 매칭**: 한글 자모 유사도 기반
        - **통합 검색**: 자모 + 의미론적 검색 조합
        """)

    # 검색 엔진 로드
    try:
        search_engine = load_search_engine()
        st.success("✅ 검색 엔진이 준비되었습니다!")
    except Exception as e:
        st.error(f"❌ 검색 엔진 로드 실패: {str(e)}")
        return

    # 검색 인터페이스
    st.header("🔍 음식점 검색")

    # 검색 입력
    query = st.text_input(
        "음식점 이름을 입력하세요",
        placeholder="예: 윤씨네, 맛있는집, 피자헛...",
        help="음식점 이름을 입력하면 다양한 방법으로 검색합니다",
    )

    # 검색 버튼
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        search_button = st.button("🔍 검색", type="primary", use_container_width=True)

    # 검색 실행
    if search_button and query.strip():
        with st.spinner("검색 중..."):
            try:
                # 검색 실행
                results = search_engine.search(
                    query=query.strip(), top_k=top_k, jamo_threshold=jamo_threshold
                )

                # 결과 표시
                if not results.empty:
                    st.success(f"✅ '{query}'에 대한 검색 결과 ({len(results)}개)")

                    # 결과 테이블
                    st.subheader("📋 검색 결과")

                    # DataFrame을 더 보기 좋게 표시
                    display_df = results.copy()

                    # 점수 컬럼이 있으면 소수점 4자리로 포맷
                    if "score" in display_df.columns:
                        display_df["score"] = display_df["score"].apply(
                            lambda x: f"{x:.4f}"
                        )

                    # 자모 점수 컬럼이 있으면 소수점 2자리로 포맷
                    if "jamo_score" in display_df.columns:
                        display_df["jamo_score"] = display_df["jamo_score"].apply(
                            lambda x: f"{x:.2f}"
                        )

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # 매칭 타입별 통계
                    st.subheader("📊 매칭 타입별 통계")
                    match_stats = results["match_type"].value_counts()

                    col1, col2, col3, col4 = st.columns(4)
                    for i, (match_type, count) in enumerate(match_stats.items()):
                        with [col1, col2, col3, col4][i]:
                            st.metric(label=match_type, value=count)

                else:
                    st.warning(f"⚠️ '{query}'에 대한 검색 결과가 없습니다.")

            except Exception as e:
                st.error(f"❌ 검색 중 오류가 발생했습니다: {str(e)}")

    elif search_button and not query.strip():
        st.warning("⚠️ 검색어를 입력해주세요.")

    # 사용 예시
    with st.expander("💡 사용 예시"):
        st.markdown("""
        ### 검색 예시:
        - **정확한 매칭**: "윤씨네" → 정확히 "윤씨네"라는 이름의 음식점
        - **부분 매칭**: "피자" → "피자헛", "도미노피자" 등 "피자"가 포함된 음식점
        - **자모 매칭**: "윤씨네" → "윤시네", "윤씨내" 등 비슷한 발음의 음식점
        - **의미론적 검색**: "맛있는 집" → "맛집", "좋은 음식점" 등 의미적으로 유사한 음식점
        """)

    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🍽️ 음식점 검색 시스템 | 다양한 검색 방법으로 원하는 음식점을 찾아보세요!
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
