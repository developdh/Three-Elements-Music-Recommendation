# Three Elements Music Recommendation

음악 추천에서 장르 기반 분류의 한계를 검증하고, 음향 특성 기반의 대안적 분류 체계를 연구하는 데이터 분석 프로젝트.

## 연구 배경

기존 음악 추천 시스템은 장르를 주요 분류 기준으로 사용하지만, 이는 과도하게 복잡하고(16+ 장르) 음향적 유사성과 잘 맞지 않을 수 있다. **"소수의 음향 특성 축만으로 음악을 분류하고 추천할 수 있지 않을까?"** 라는 질문에서 출발했다.

## 연구 과정 및 결과

### Experiment V1: 3개 PCA 주성분이면 충분한가?

> `python3 src/main.py`

FMA 데이터셋(49,598곡, 518개 오디오 피처)에 PCA를 적용하여 3개 주성분의 설명력을 검증.

| 지표 | 결과 | 기준 | 판정 |
|------|------|------|------|
| 누적 분산 설명률 | **23.49%** | >= 80% | FAIL |
| k-NN overlap | **1.55%** | >= 70% | FAIL |
| 80% 달성 최소 차원 | **85개** | - | - |

**결론:** 518개 raw 피처에 PCA를 그대로 적용하면 3개 차원으로는 부족하다. 각 기본 피처당 7개 통계량(mean, std, min, max, median, skew, kurtosis)이 있어 중복이 심하다.

### Experiment V2: 피처 엔지니어링으로 3차원을 살릴 수 있는가?

> `python3 src/experiment_v2.py`

3가지 접근법으로 피처 공간을 정제한 후 3차원 축소를 시도.

| 접근법 | 3PC 분산 설명률 | Silhouette | k-NN Overlap |
|--------|---------------|------------|-------------|
| Raw 518 PCA (V1) | 23.5% | -0.097 | 1.6% |
| Mean-Only PCA (74개) | 35.7% | -0.205 | 1.1% |
| **Group Aggregation PCA (6그룹)** | **65.4%** | -0.188 | 0.7% |
| Domain 3-Axis | N/A | -0.262 | 1.0% |

**핵심 발견:** 518개 전체 피처로도 장르 기반 silhouette score가 **음수(-0.097)**. 이는 **장르 자체가 오디오 피처 공간에서 의미 있는 클러스터를 형성하지 않는다**는 뜻이다.

### Experiment V3: 장르 대신 음향 기반 자연 클러스터

> `python3 src/experiment_v3.py`

장르 레이블을 버리고 비지도 학습(K-Means)으로 음악의 자연 그룹을 발견.

| 비교 기준 | 자연 클러스터 (k=2) | 장르 (16개) |
|-----------|-------------------|------------|
| **Silhouette (3D)** | **+0.247** | -0.188 |
| **k-NN Purity** | **97.7%** | 24.9% |

**자연 클러스터의 특성:**
- **Cluster 0 (59%)**: 낮은 에너지 — Classical(82%), Folk(78%), Rock(73%), Old-Time(95%)
- **Cluster 1 (41%)**: 높은 에너지 — Electronic(65%), Hip-Hop(69%), Soul-RnB(74%)

**핵심 결론:**
1. 음악의 자연적 구조는 16개 장르가 아니라 **2개의 음향 클러스터**로 더 잘 설명된다
2. 자연 클러스터는 3D 공간에서 **깨끗하게 분리**된다 (silhouette +0.247 vs -0.188)
3. 장르 레이블과 자연 클러스터의 상관관계는 **거의 없다** (ARI = 0.037)
4. **장르 기반 분류는 음향적 유사성과 맞지 않으며**, 연속적 음향 특성 공간이 더 적절하다

### Experiment V4: 클러스터 수를 늘려도 장르보다 나은가?

> `python3 src/experiment_v4.py`

k=2가 너무 단순할 수 있으므로, k=2~10까지 늘려가며 장르 대비 우위가 유지되는지 검증.

| k | Silhouette | k-NN Purity | vs Genre Sil | vs Genre Purity |
|---|-----------|-------------|-------------|----------------|
| 2 | +0.247 | 97.7% | **+0.435** | **+72.8%p** |
| 3 | +0.237 | 96.3% | **+0.425** | **+71.4%p** |
| 5 | +0.217 | 88.5% | **+0.405** | **+63.6%p** |
| 10 | +0.135 | 70.2% | **+0.323** | **+45.3%p** |
| Genre (16) | -0.188 | 24.9% | baseline | baseline |

**핵심 결론:**
- **k=2~10 모두** 장르보다 우수 — 클러스터 우위가 모든 세분화 수준에서 유지됨
- k=10에서도 silhouette +0.135 vs 장르 -0.188 → **여전히 압도적**
- 클러스터-장르 상관관계(ARI)는 0.04~0.06으로 **k에 상관없이 거의 무관**
- 이는 장르와 음향 유사성이 근본적으로 다른 축임을 확증

### Experiment V5: 3축 추천 vs 장르 기반 추천

> `python3 src/experiment_v5.py`

3개 축(group-aggregated PCA)으로 추천 시스템을 만들고 장르 기반 추천과 비교.

| 지표 | Genre-Based | 3-Axis (Ours) | Oracle (518D) | 승자 |
|------|-------------|---------------|---------------|------|
| Oracle Overlap | 29.3% | 0.7% | 100% | Genre |
| Intra-list Similarity | 0.572 | 0.150 | 0.565 | Genre |
| **Genre Diversity** | 10.0% | **46.7%** | 34.9% | **3-Axis** |
| **Catalog Coverage** | 29.2% | **33.2%** | 24.3% | **3-Axis** |

**핵심 분석:**
- 3-Axis는 **정밀도에서 장르에 뒤지지만**, 장르 다양성에서 **4.7배** 우수
- 장르 기반은 같은 장르 안에서만 추천 → 버블 효과 (diversity 10%)
- 3-Axis는 장르 경계를 넘어 **음향적으로 유사한 곡을 발견** → 세렌디피티(serendipity)
- Oracle조차 장르 다양성이 34.9% → **음향적 유사성은 장르를 넘나든다**는 증거
- 즉, **3축은 "정확한 추천"보다 "새로운 발견"에 적합**한 보완적 시스템

## 연구 종합 결론

1. **장르 분류는 오버엔지니어링이 맞다** — 16개 장르는 음향 공간에서 구분되지 않음 (silhouette -0.19)
2. **음악의 자연 구조는 장르보다 단순하다** — 2개의 에너지 기반 클러스터가 장르보다 231% 더 잘 분리
3. **3개 축으로 줄이는 것은 가능하지만 목적에 따라 다르다**:
   - 정밀 추천 (같은 느낌의 곡): 3개 축으로는 부족, 더 많은 차원 필요
   - 탐색/발견 (새로운 곡 발견): 3개 축이 장르보다 4.7배 더 다양한 추천 제공
4. **가장 유망한 접근**: 3축(탐색) + 세부 피처(정밀) 하이브리드 시스템

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
# V1: PCA 분석 (3개 주성분 충분성 검증)
python3 src/main.py

# V2: 피처 엔지니어링 비교 실험
python3 src/experiment_v2.py

# V3: 비지도 클러스터링 (자연 음악 그룹 발견)
python3 src/experiment_v3.py

# V4: 클러스터 확장성 검증 (k=2~10 vs 장르)
python3 src/experiment_v4.py

# V5: 추천 시스템 비교 (3축 vs 장르 vs Oracle)
python3 src/experiment_v5.py
```

## Output

### V1 (`src/main.py`)
- `output/scree_plot.png` — 분산 설명률 (개별 + 누적)
- `output/3d_scatter.png` — 장르별 3D 분포
- `output/loadings_heatmap.png` — 주성분별 피처 기여도

### V2 (`src/experiment_v2.py`)
- `output/v2_comparison.png` — 3가지 접근법 비교

### V3 (`src/experiment_v3.py`)
- `output/v3_optimal_k.png` — 최적 클러스터 수 분석 (Elbow + Silhouette)
- `output/v3_clusters_vs_genres.png` — 자연 클러스터 vs 장르 3D 비교
- `output/v3_cluster_profiles.png` — 클러스터별 음향 특성 프로필

### V4 (`src/experiment_v4.py`)
- `output/v4_scalability.png` — k=2~10 Silhouette/Purity/ARI 비교 (vs 장르 baseline)
- `output/v4_clusters_k{N}.png` — 최적 k의 3D 클러스터 분포

### V5 (`src/experiment_v5.py`)
- `output/v5_recommendation_comparison.png` — 추천 성능 비교 (Genre vs 3-Axis vs Oracle)
- `output/v5_genre_precision.png` — 장르 정밀도 비교

## Project Structure

```
Three-Elements-Music-Recommendation/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py              # 설정값 중앙 관리
│   ├── main.py                # V1: PCA 분석
│   ├── experiment_v2.py       # V2: 피처 엔지니어링 비교
│   ├── experiment_v3.py       # V3: 비지도 클러스터링
│   ├── experiment_v4.py       # V4: 클러스터 확장성 검증
│   ├── experiment_v5.py       # V5: 추천 시스템 성능 비교
│   ├── data_loader.py         # FMA 데이터 로드/전처리
│   ├── pca_analysis.py        # PCA 분석 모듈
│   ├── visualization.py       # 시각화 모듈
│   └── evaluation.py          # k-NN 평가 모듈
├── data/fma_metadata/         # FMA 데이터 (gitignored)
└── output/                    # 생성된 그래프 (gitignored)
```
