# Three Elements Music Recommendation

음악 간 유사성을 정의하는 최소 차원 수를 연구하는 데이터 분석 프로젝트.

**핵심 연구 질문:** "음악 간 유사성을 측정하는 데 3개 차원이면 충분한가?"

FMA(Free Music Archive) 데이터셋의 오디오 피처에 PCA를 적용하여, 3개 주성분의 누적 분산 설명률(≥80%)과 k-NN 추천 overlap(≥70%)으로 검증합니다.

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

1. [FMA GitHub](https://github.com/mdeff/fma)에서 메타데이터를 다운로드합니다:
   - `fma_metadata.zip` 다운로드 (342MB)
2. 압축 해제 후 아래 파일을 `data/fma_metadata/`에 배치합니다:
   - `features.csv`
   - `tracks.csv`

```bash
mkdir -p data/fma_metadata
# fma_metadata.zip 압축 해제 후 features.csv, tracks.csv를 data/fma_metadata/에 복사
```

## Run

```bash
python3 src/main.py
```

## Output

- `output/scree_plot.png` — 개별 + 누적 분산 설명률
- `output/3d_scatter.png` — 3개 주성분 공간에서 장르별 분포
- `output/loadings_heatmap.png` — 주성분별 피처 기여도
- 콘솔에 분석 결과 요약 출력
