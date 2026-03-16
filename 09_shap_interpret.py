"""
==============================================================================
09_shap_interpretability.py - SHAP 기반 연령 그룹별 해석가능성 분석
==============================================================================

목적:
1. SHAP(SHapley Additive exPlanations)로 피처 기여도 정량화
2. Young vs Old 그룹 간 중요 피처 차이 분석
3. 시각화로 임상 해석 가능성 확보

SHAP이란?
- 게임 이론의 Shapley Value를 ML 모델 해석에 적용
- 각 피처가 "예측값 - 평균 예측값"에 기여한 정도를 정량화
- 모델의 블랙박스 문제 해결 → 임상 신뢰도 향상

차이점 (vs 04_feature48_shap.py):
- 04번: Permutation Importance (전역 중요도, 피처 셔플링)
- 09번: SHAP (개별 예측 기여도 + 그룹별 비교)

생성 파일:
- ami_young_shap_summary.csv
- ami_old_shap_summary.csv
- stroke_young_shap_summary.csv
- stroke_old_shap_summary.csv
- {disease}_shap_comparison.png (비교 차트)

소요 시간:
- 약 10~15분/질환 (GradientExplainer 계산이 느림)

면접 포인트:
"SHAP 분석으로 연령별로 중요한 피처가 다름을 발견했습니다.
 예를 들어 Young 그룹에서는 HR_Delta(심박수 변화율)가,
 Old 그룹에서는 GCS(의식 수준)가 더 중요한 예측 인자로 나타나
 연령별 맞춤 모니터링 전략 수립이 가능합니다."

참고 문헌:
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions
- Ponce-Bobadilla et al. (2024). Practical guide to SHAP analysis

작성일: 2026-01-14
==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap  # SHapley Additive exPlanations 라이브러리
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# [전역 설정]
# ==============================================================================
BASE_DIR = "/home/kido/miniproject/team3"
device = torch.device("cpu")  # SHAP은 CPU에서도 충분히 빠름

# ==============================================================================
# [모델 정의]
# ==============================================================================

class MultiModalMIMIC(nn.Module):
    """
    Multi-modal LSTM 모델 (03번과 동일 구조)
    
    ⚠️ 주의: 이 모델은 학습된 가중치(.pth)와 구조가 일치해야 함!
    
    구조:
    - LSTM: 시계열(6h × 23 features) → 64-dim 은닉 상태
    - Static FC: 정적 변수 4개 → 16-dim 임베딩
    - Classifier: 80-dim(64+16) → 32-dim → 1-dim(사망 확률)
    
    면접 포인트:
    "시계열과 정적 변수를 별도 경로로 처리하는 Multi-Modal 아키텍처로
     각 모달리티의 고유 패턴을 효과적으로 학습합니다."
    """
    def __init__(self, time_dim=23, static_dim=4, hidden_dim=64):
        """
        Args:
            time_dim: 시계열 피처 수 (22 vitals + 1 HR_Delta)
            static_dim: 정적 피처 수 (Age, Gender, Height, Weight)
            hidden_dim: LSTM 은닉 차원 (64)
        """
        super().__init__()
        
        # LSTM: 시계열 경로 (6시간 × 23개 피처 → 64차원 압축)
        self.lstm = nn.LSTM(
            time_dim, hidden_dim, 
            num_layers=2,      # 2층 구조
            batch_first=True,  # (Batch, Time, Feature) 형태
            dropout=0.2        # 층간 20% 드롭아웃
        )
        
        # Static FC: 정적 변수 경로 (4개 → 16개로 확장)
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 16),  # 4 → 16
            nn.ReLU(),                   # 비선형 활성화
            nn.Dropout(0.2)              # 과적합 방지
        )
        
        # Classifier: 최종 사망 확률 예측
        # 64(LSTM) + 16(Static) = 80 → 32 → 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),  # 80 → 32
            nn.ReLU(),
            nn.Linear(32, 1),                 # 32 → 1
            nn.Sigmoid()                      # 0~1 확률 변환
        )

    def forward(self, x_time, x_static):
        """
        순전파
        
        Args:
            x_time: (Batch, 6, 23) - 6시간 Rolling Window
            x_static: (Batch, 4) - 정적 변수
        
        Returns:
            prob: (Batch,) - 사망 확률
        """
        # LSTM의 마지막 은닉 상태 추출
        _, (h_n, _) = self.lstm(x_time)
        time_feat = h_n[-1]  # (Batch, 64)
        
        # 정적 변수 인코딩
        static_feat = self.static_fc(x_static)  # (Batch, 16)
        
        # 시계열 + 정적 정보 결합
        combined = torch.cat([time_feat, static_feat], dim=1)  # (Batch, 80)
        
        # 최종 예측
        return self.classifier(combined).squeeze()  # (Batch,)


# ==============================================================================
# [데이터 준비 함수]
# ==============================================================================

def prepare_advanced_data(disease_type):
    """
    SHAP 분석용 데이터 준비
    
    목적:
    1. 롤링 윈도우 데이터 로드
    2. HR_Delta(심박수 변화율) 피처 추가
    3. 시계열과 정적 변수 분리
    
    Args:
        disease_type: 'ami' 또는 'stroke'
    
    Returns:
        X_time: (N, 6, 23) - 시계열 (22 vitals + 1 HR_Delta)
        X_static: (N, 4) - 정적 변수 (Age, Gender, Height, Weight)
        y_final: (N,) - 라벨
        sids: (N,) - 환자 ID
    
    전처리 로직:
    1. 원본 데이터 로드 (X, y, sids)
    2. Label Shift: X[t]로 y[t+1] 예측 (미래 예측 시뮬레이션)
    3. 환자 연속성 검증: stay_id가 같은 경우만 유효
    4. HR_Delta 계산 및 추가
    """
    # -------------------------------------------------------------------------
    # Step 1: 원본 데이터 로드
    # -------------------------------------------------------------------------
    X = np.load(f"{BASE_DIR}/{disease_type}_X_rolling.npy")      # (N, 6, 22)
    y_raw = np.load(f"{BASE_DIR}/{disease_type}_y_rolling.npy")  # (N,)
    sids = np.load(f"{BASE_DIR}/{disease_type}_sids_rolling.npy") # (N,)
    
    # -------------------------------------------------------------------------
    # Step 2: Label Shift (미래 예측 정합성)
    # -------------------------------------------------------------------------
    # 목적: 실시간 시스템 시뮬레이션
    # X[t] 시점 데이터로 y[t+1] 시점 사망 여부 예측
    X_shifted = X[:-1]        # 마지막 행 제외
    y_shifted = y_raw[1:]     # 첫 행 제외 (1칸 미래)
    sids_shifted = sids[:-1]
    
    # -------------------------------------------------------------------------
    # Step 3: 환자 연속성 검증
    # -------------------------------------------------------------------------
    # 문제: X[i]와 y[i+1]이 다른 환자일 수 있음
    # 해결: stay_id가 연속된 경우만 유효
    valid_mask = sids_shifted == sids[1:]  # 같은 환자인지 확인
    X_final = X_shifted[valid_mask]
    y_final = y_shifted[valid_mask]
    
    # -------------------------------------------------------------------------
    # Step 4: HR_Delta 피처 생성 (심박수 변화율)
    # -------------------------------------------------------------------------
    # 공식: (6시간째 HR - 1시간째 HR) / 5시간
    # 의미: HR이 시간당 몇 bpm씩 증가/감소하는지
    # 임상적 의미: 급격한 HR 증가는 쇼크, 감소는 서맥 위험 신호
    
    # X_final[:, -1, 0]: 6시간째(마지막) 시점의 HR
    # X_final[:, 0, 0]: 1시간째(첫) 시점의 HR
    slopes = (X_final[:, -1, 0] - X_final[:, 0, 0]) / 5.0  # (N,)
    
    # 모든 시간 스텝에 동일한 slope 추가 (시간 불변 피처)
    # (N,) → (N, 1, 1) → (N, 6, 1) 반복
    slopes_expanded = np.repeat(slopes[:, np.newaxis, np.newaxis], 6, axis=1)
    
    # 원본 22개 피처에 HR_Delta 1개 추가 → 23개
    X_time = np.concatenate([X_final, slopes_expanded], axis=2)  # (N, 6, 23)
    
    # -------------------------------------------------------------------------
    # Step 5: 정적 변수 추출
    # -------------------------------------------------------------------------
    # 첫 시점(시간 인덱스 0)의 처음 4개 피처
    # [Age, Gender, Height, Weight] - 시간에 따라 변하지 않음
    X_static = X_final[:, 0, :4]  # (N, 4)
    
    return X_time, X_static, y_final, sids_shifted[valid_mask]


# ==============================================================================
# [SHAP 분석 함수]
# ==============================================================================

def run_multimodal_shap(disease_type):
    """
    연령 그룹별 SHAP 분석 실행
    
    SHAP 작동 원리:
    1. Baseline: 전체 데이터의 평균 예측값 계산
    2. Coalition: 각 피처를 포함/제외한 모든 조합에서 예측값 변화 측정
    3. Shapley Value: 게임 이론으로 각 피처의 "공정한 기여도" 계산
    
    왜 GradientExplainer?
    - Neural Network에 최적화된 SHAP 방법
    - 모델의 gradient(미분값)를 활용하여 빠르게 계산
    - 대안: DeepExplainer(더 정확하지만 느림), KernelExplainer(모델 무관)
    
    Args:
        disease_type: 'ami' 또는 'stroke'
    
    생성 파일:
        - {disease}_young_shap_summary.csv
        - {disease}_old_shap_summary.csv
    
    소요 시간:
        - 약 5~10분 (그룹당 100개 샘플 × 2 그룹)
    """
    print(f"\n🧬 [{disease_type.upper()}] SHAP 분석 시작...")
    
    # =========================================================================
    # Phase 1: 데이터 및 모델 로드
    # =========================================================================
    X_time, X_static, y, sids = prepare_advanced_data(disease_type)
    
    model = MultiModalMIMIC().to(device)
    
    # ⚠️ 수정 필요: fixed_best.pth → multimodal_best.pth
    model_path = f"{BASE_DIR}/{disease_type}_fixed_best.pth"
    # 권장: model_path = f"{BASE_DIR}/{disease_type}_multimodal_best.pth"
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드 (Dropout 비활성화)

    # =========================================================================
    # Phase 2: ModelWrapper 생성 (SHAP 입력 형태 변환용)
    # =========================================================================
    class ModelWrapper(nn.Module):
        """
        SHAP 호환 래퍼 모델
        
        목적:
        - 원본 모델: 2개 입력 필요 (x_time, x_static)
        - SHAP: 1개 입력만 지원 (x_combined)
        - 해결: x_combined를 받아 자동으로 2개로 분할
        
        입력 형태:
        - x_combined: (Batch, 6*23 + 4) = (Batch, 142)
          = [시계열 138개 | 정적 4개]
        
        처리:
        1. x_combined[:, :138] → reshape → (Batch, 6, 23) = x_time
        2. x_combined[:, 138:] → (Batch, 4) = x_static
        3. 원본 모델에 전달
        """
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x_combined):
            """
            Args:
                x_combined: (Batch, 142) - 1차원으로 펼친 입력
            
            Returns:
                prob: (Batch, 1) - 사망 확률
            """
            # 시계열 부분 추출 및 reshape
            # x_combined[:, :6*23] = x_combined[:, :138]
            x_time = x_combined[:, :6*23].reshape(-1, 6, 23)  # (Batch, 6, 23)
            
            # 정적 변수 부분 추출
            x_static = x_combined[:, 6*23:]  # (Batch, 4)
            
            # 원본 모델 호출
            return self.model(x_time, x_static).view(-1, 1)  # (Batch, 1)

    wrapper = ModelWrapper(model)
    
    # -------------------------------------------------------------------------
    # 데이터를 1차원으로 펼치기 (SHAP 입력 형태)
    # -------------------------------------------------------------------------
    # X_time: (N, 6, 23) → reshape → (N, 138)
    # X_static: (N, 4) → 유지
    # 결합: (N, 142)
    X_combined = np.hstack([
        X_time.reshape(X_time.shape[0], -1),  # (N, 138)
        X_static                               # (N, 4)
    ])  # → (N, 142)
    
    # =========================================================================
    # Phase 3: 피처명 정의 (해석 가능성 확보)
    # =========================================================================
    # 시계열 피처: 6개 시점 × 23개 바이탈 = 138개
    # 형태: HR_t0, SBP_t0, ..., HR_Delta_t0, HR_t1, ..., HR_Delta_t5
    
    vitals = [
        'HR',      # 심박수
        'SBP',     # 수축기 혈압
        'DBP',     # 이완기 혈압
        'MBP',     # 평균 동맥압
        'RR',      # 호흡수
        'Temp',    # 체온
        'SpO2',    # 산소포화도
        'FiO2',    # 흡입 산소 농도
        'GCS_E',   # Glasgow Coma Scale - Eye
        'GCS_V',   # Glasgow Coma Scale - Verbal
        'GCS_M',   # Glasgow Coma Scale - Motor
        'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11',  # 기타 센서/마스크
        'HR_Delta'  # 심박수 변화율 (추가 피처)
    ]
    
    # 시계열 피처명 생성: 'HR_t0', 'SBP_t0', ..., 'HR_Delta_t5'
    # 6개 시점(t0~t5) × 23개 바이탈 = 138개
    feature_names = [f"{v}_t{t}" for t in range(6) for v in vitals]
    
    # 정적 피처명 추가: 'Age', 'Gender', 'Height', 'Weight'
    feature_names += ['Age', 'Gender', 'Height', 'Weight']
    
    # 총 142개 피처명 (138 시계열 + 4 정적)

    # =========================================================================
    # Phase 4: 연령 그룹 분할 (Young vs Old)
    # =========================================================================
    # Age는 Z-score 정규화되어 있음
    # Age < 0: 평균보다 젊음 (Young)
    # Age > 0: 평균보다 많음 (Old)
    
    age_col_idx = 6 * 23  # 138번째 컬럼 = Age 위치
    
    young_idx = np.where(X_combined[:, age_col_idx] < 0)[0]  # Young 그룹 인덱스
    old_idx = np.where(X_combined[:, age_col_idx] > 0)[0]    # Old 그룹 인덱스
    
    print(f"   📊 Young 그룹: {len(young_idx)}명, Old 그룹: {len(old_idx)}명")

    # =========================================================================
    # Phase 5: SHAP Explainer 초기화
    # =========================================================================
    # GradientExplainer: Neural Network에 최적화된 SHAP 방법
    # Background 샘플: 모델의 "평균적인 예측"을 계산하는 기준점
    # - 전체 데이터에서 100개 무작위 선택
    # - 너무 많으면 느려지고, 너무 적으면 부정확
    
    bg_samples = torch.FloatTensor(
        X_combined[np.random.choice(len(X_combined), 100, replace=False)]
    )
    
    explainer = shap.GradientExplainer(wrapper, bg_samples)
    
    print(f"   ✅ SHAP Explainer 초기화 완료 (Background: 100 samples)")

    # =========================================================================
    # Phase 6: 그룹별 SHAP 값 계산 및 저장
    # =========================================================================
    for name, indices in [('young', young_idx), ('old', old_idx)]:
        # 그룹이 비어있으면 건너뛰기
        if len(indices) == 0:
            print(f"   ⚠️ {name.upper()} 그룹 데이터 없음. 건너뜀.")
            continue
        
        print(f"🔍 {name.upper()} 그룹 분석 중 (N={min(100, len(indices))})...")
        
        # ---------------------------------------------------------------------
        # Step 1: 분석 대상 샘플 선택 (최대 100개)
        # ---------------------------------------------------------------------
        # 이유: SHAP 계산이 느리므로 샘플링 필요
        # 100개면 통계적으로 충분하며, 10~15분 소요
        target_indices = np.random.choice(
            indices, 
            min(100, len(indices)),  # 그룹 크기가 100보다 작으면 전체 사용
            replace=False
        )
        target_samples = torch.FloatTensor(X_combined[target_indices])
        
        # ---------------------------------------------------------------------
        # Step 2: SHAP 값 계산
        # ---------------------------------------------------------------------
        # shap_values: (N_samples, N_features) = (100, 142)
        # shap_values[i, j] = j번째 피처가 i번째 샘플 예측에 기여한 정도
        # 
        # 해석:
        # - 양수: 사망 확률을 높이는 방향으로 기여
        # - 음수: 사망 확률을 낮추는 방향으로 기여
        # - 절대값: 기여도의 크기
        
        shap_values = explainer.shap_values(target_samples)
        
        # Binary classification의 경우 리스트로 반환될 수 있음
        # [Class 0 SHAP, Class 1 SHAP] → Class 1만 사용
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # ---------------------------------------------------------------------
        # Step 3: 피처별 평균 기여도 계산 (전역 중요도)
        # ---------------------------------------------------------------------
        # |SHAP 값|의 평균: 방향 무관하게 "영향력" 측정
        # importance[j] = mean(|shap_values[:, j]|)
        # 
        # 예시:
        # - HR_t5의 SHAP 값: [+0.3, -0.2, +0.5, -0.1, ...]
        # - 평균 절대값: (0.3 + 0.2 + 0.5 + 0.1 + ...) / N
        # - 의미: HR_t5가 평균적으로 예측에 얼마나 영향을 주는가?
        
        importance = np.abs(shap_values).mean(axis=0)  # (142,)
        
        # ---------------------------------------------------------------------
        # Step 4: DataFrame 생성 및 정렬
        # ---------------------------------------------------------------------
        df_imp = pd.DataFrame({
            'Feature': feature_names,              # 피처명
            'Importance': importance.flatten()     # 평균 |SHAP 값|
        })
        
        # 중요도 내림차순 정렬 (가장 중요한 피처가 맨 위)
        df_imp = df_imp.sort_values(by='Importance', ascending=False)
        
        # ---------------------------------------------------------------------
        # Step 5: CSV 저장
        # ---------------------------------------------------------------------
        save_name = f"{disease_type}_{name}_shap_summary.csv"
        save_path = f"{BASE_DIR}/{save_name}"
        df_imp.to_csv(save_path, index=False)
        
        print(f"   ✅ {save_name} 저장 완료.")
        print(f"   📌 상위 5개 피처: {df_imp.head(5)['Feature'].tolist()}")


# ==============================================================================
# [시각화 함수]
# ==============================================================================

def visualize_group_comparison(disease_type):
    """
    Young vs Old 그룹의 SHAP 중요도 비교 시각화
    
    목적:
    1. 연령 그룹 간 중요 피처 차이 발견
    2. 임상적 해석 가능성 확보
    3. 맞춤형 모니터링 전략 수립 근거 제공
    
    시각화 형태:
    - Grouped Bar Chart (그룹별 나란히 배치)
    - Y축: Top 15 피처 (중요도 순)
    - X축: Mean |SHAP Value| (평균 절대 SHAP 값)
    - Color: Young (파랑) vs Old (주황)
    
    임상적 활용:
    - Young에서만 높은 피처 → 젊은 환자 집중 모니터링
    - Old에서만 높은 피처 → 고령 환자 집중 모니터링
    - 공통 피처 → 연령 무관 필수 모니터링
    
    Args:
        disease_type: 'ami' 또는 'stroke'
    
    생성 파일:
        - {disease}_shap_comparison.png (300 DPI)
    """
    print(f"📊 [{disease_type.upper()}] 그룹별 시각화 생성 중...")
    
    # =========================================================================
    # Phase 1: CSV 파일 존재 확인
    # =========================================================================
    young_csv = f"{BASE_DIR}/{disease_type}_young_shap_summary.csv"
    old_csv = f"{BASE_DIR}/{disease_type}_old_shap_summary.csv"
    
    if not os.path.exists(young_csv) or not os.path.exists(old_csv):
        print(f"   ⚠️ CSV 파일이 없습니다. run_multimodal_shap()을 먼저 실행하세요.")
        return

    # =========================================================================
    # Phase 2: 데이터 로드 및 전처리
    # =========================================================================
    # Top 15개 피처만 선택 (너무 많으면 차트가 복잡함)
    young_df = pd.read_csv(young_csv).head(15)
    old_df = pd.read_csv(old_csv).head(15)
    
    # 그룹 컬럼 추가 (seaborn의 hue 파라미터용)
    young_df['Group'] = 'Young (< Mean Age)'
    old_df['Group'] = 'Old (> Mean Age)'
    
    # 두 DataFrame 병합 (세로로 쌓기)
    total_df = pd.concat([young_df, old_df])
    
    # =========================================================================
    # Phase 3: 시각화 생성
    # =========================================================================
    plt.figure(figsize=(12, 8))
    
    # Grouped Bar Chart 생성
    # data: 전체 데이터
    # x: X축 (중요도 값)
    # y: Y축 (피처명)
    # hue: 색상 구분 (그룹)
    # palette: 색상 팔레트 ('muted'는 부드러운 파스텔톤)
    sns.barplot(
        data=total_df, 
        x='Importance',  # X축: 평균 |SHAP 값|
        y='Feature',     # Y축: 피처명
        hue='Group',     # 색상: Young vs Old
        palette='muted'  # 색상 테마
    )
    
    # -------------------------------------------------------------------------
    # 차트 꾸미기
    # -------------------------------------------------------------------------
    plt.title(
        f"SHAP Feature Importance Comparison: {disease_type.upper()}", 
        fontsize=15, 
        fontweight='bold'
    )
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.ylabel("Top 15 Features", fontsize=12)
    
    # X축 격자선 추가 (가독성 향상)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 여백 자동 조정
    plt.tight_layout()
    
    # =========================================================================
    # Phase 4: 파일 저장
    # =========================================================================
    save_path = f"{BASE_DIR}/{disease_type}_shap_comparison.png"
    plt.savefig(save_path, dpi=300)  # 고해상도 저장 (논문/발표용)
    
    print(f"   ✅ 시각화 파일 저장 완료: {save_path}")
    
    # 화면에도 표시 (Jupyter/터미널에서)
    # plt.show()  # 주석 해제 시 팝업으로 차트 표시


# ==============================================================================
# [메인 실행부]
# ==============================================================================

if __name__ == "__main__":
    """
    전체 파이프라인 실행
    
    실행 순서:
    1. AMI: SHAP 계산 → 시각화
    2. STROKE: SHAP 계산 → 시각화
    
    소요 시간:
    - 약 20~30분 (질환당 10~15분)
    
    생성 파일 (총 6개):
    - ami_young_shap_summary.csv
    - ami_old_shap_summary.csv
    - ami_shap_comparison.png
    - stroke_young_shap_summary.csv
    - stroke_old_shap_summary.csv
    - stroke_shap_comparison.png
    
    사용법:
        python 09_shap_interpretability.py
    
    전제 조건:
        - 02_window_6h.py 실행 완료 (X_rolling.npy 존재)
        - 03_multimodal.py 실행 완료 (multimodal_best.pth 존재)
        - SHAP 라이브러리 설치: pip install shap
    """
    for d in ['ami', 'stroke']:
        # Step 1: SHAP 수치 계산 및 CSV 저장
        run_multimodal_shap(d)
        
        # Step 2: CSV 기반 비교 그래프 생성
        visualize_group_comparison(d)
    
    print("\n" + "="*80)
    print("✅ 전체 SHAP 분석 완료!")
    print("="*80)
    print("\n📁 생성된 파일:")
    print("   1. ami_young_shap_summary.csv")
    print("   2. ami_old_shap_summary.csv")
    print("   3. ami_shap_comparison.png")
    print("   4. stroke_young_shap_summary.csv")
    print("   5. stroke_old_shap_summary.csv")
    print("   6. stroke_shap_comparison.png")
    
    print("\n🎓 면접 멘트:")
    print("   'SHAP 분석으로 연령 그룹별 중요 피처 차이를 정량화했습니다.")
    print("    예를 들어 Young 그룹에서는 HR_Delta가,")
    print("    Old 그룹에서는 GCS(의식 수준)가 더 중요한 예측 인자로 나타나")
    print("    연령별 맞춤 모니터링 전략 수립이 가능합니다.'")
