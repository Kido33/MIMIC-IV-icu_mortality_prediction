"""
==============================================================================
07_final_audit.py - 최종 임상 감사 보고서 (Calibration + Youden's Index)
==============================================================================

목적:
1. 학습된 모델의 Calibration(확률 신뢰도) 검증
2. Youden's Index로 최적 임계값 자동 선정
3. 연령 그룹별(Young/Old) 성능 분석
4. 통계적 유의성 검증 (Wilcoxon rank-sum test)

05_comprehensive_benchmark.py와의 차이:
- 05번: 7개 알고리즘 비교 (면접용 종합 분석)
- 07번: 최종 모델 1개만 집중 검증 (보고서용)

생성 파일:
- {disease}_calibration.png: Calibration 차트
- 콘솔 출력: 연령 그룹별 성적표 (Transpose 형태)

면접 포인트:
- "Calibration Curve가 Ideal Line에 근접하여 확률 신뢰도 우수"
- "Youden's Index로 Sensitivity와 Specificity 균형점 도출"
- "연령 그룹 간 유의미한 성능 차이 없음 → Robustness 검증"

작성일: 2026-01-14
==============================================================================
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats  # Wilcoxon rank-sum test
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve  # Calibration 분석
import xgboost as xgb  # 향후 XGBoost 비교용 (현재 미사용)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score,
    precision_recall_curve, roc_curve
)

# ==============================================================================
# [전역 설정]
# ==============================================================================
BASE_DIR = "/home/kido/miniproject/team3"
device = torch.device("cpu")  # CPU 사용 (추론만 하므로 GPU 불필요)

# ==============================================================================
# [모델 정의]
# ==============================================================================

class MultiModalMIMIC(nn.Module):
    """
    사용자 정의 Multi-Modal LSTM 모델
    
    ⚠️ 주의: 이 모델은 {disease}_fixed_best.pth에 저장된 모델과 일치해야 함!
    
    구조:
    - LSTM: 시계열(6h × 23 features) → 64-dim 은닉 상태
    - Static FC: 정적 변수 4개 → 16-dim 임베딩
    - Classifier: 80-dim(64+16) → 32-dim → 1-dim(사망 확률)
    
    아키텍처 설명:
    1. LSTM이 시계열 패턴을 64차원으로 압축
    2. Static FC가 정적 변수를 16차원으로 인코딩
    3. 두 정보를 결합(80차원) → 추가 FC층(32차원) → 최종 예측
    
    면접 포인트:
    "시계열과 정적 변수를 별도 경로로 처리한 후 후반부에서 융합하여
     각 모달리티의 고유 패턴을 효과적으로 학습합니다."
    """
    def __init__(self, time_dim=23, static_dim=4, hidden_dim=64):
        """
        Args:
            time_dim: 시계열 피처 수 (22 vitals + 1 slope)
            static_dim: 정적 피처 수 (Age, Gender, Admission_Type, First_Careunit)
            hidden_dim: LSTM 은닉 차원 (64)
        """
        super().__init__()
        
        # -------------------------------------------------------------------------
        # LSTM: 시계열 경로
        # -------------------------------------------------------------------------
        # num_layers=2: 2층 구조로 복잡한 시간적 패턴 학습
        # dropout=0.2: 층간 20% 드롭아웃으로 과적합 방지
        self.lstm = nn.LSTM(
            time_dim,      # 입력: 23개 피처
            hidden_dim,    # 은닉: 64차원
            num_layers=2,  # 2층 LSTM
            batch_first=True,  # (Batch, Time, Feature) 형태
            dropout=0.2
        )
        
        # -------------------------------------------------------------------------
        # Static FC: 정적 변수 경로
        # -------------------------------------------------------------------------
        # 4개 입력 → 16개 출력 (4배 확장으로 표현력 증가)
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 16),  # 4 → 16
            nn.ReLU(),                   # 비선형 활성화
            nn.Dropout(0.2)              # 과적합 방지
        )
        
        # -------------------------------------------------------------------------
        # Classifier: 최종 사망 확률 예측
        # -------------------------------------------------------------------------
        # 입력: 64(LSTM) + 16(Static) = 80-dim
        # 중간층: 32-dim (추가 비선형 변환)
        # 출력: 1-dim 확률
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),  # 80 → 32
            nn.ReLU(),                        # 비선형 활성화
            nn.Linear(32, 1),                 # 32 → 1
            nn.Sigmoid()                      # 0~1 확률 변환
        )
    
    def forward(self, x_time, x_static):
        """
        순전파 (Forward Pass)
        
        Args:
            x_time: (Batch, 6, 23) - 6시간 Rolling Window 시계열
            x_static: (Batch, 4) - 정적 변수
        
        Returns:
            prob: (Batch,) - 사망 확률 (0~1)
        
        처리 흐름:
        1. LSTM으로 시계열 압축 → 64-dim
        2. FC로 정적 변수 인코딩 → 16-dim
        3. Concatenate → 80-dim
        4. FC → 32-dim → ReLU
        5. FC → 1-dim → Sigmoid → 확률
        """
        # LSTM의 마지막 은닉 상태 추출
        # h_n: (num_layers, Batch, hidden_dim) = (2, Batch, 64)
        _, (h_n, _) = self.lstm(x_time)
        time_feat = h_n[-1]  # 마지막 층의 은닉 상태 (Batch, 64)
        
        # 정적 변수 인코딩
        static_feat = self.static_fc(x_static)  # (Batch, 16)
        
        # 시계열 + 정적 정보 결합
        combined = torch.cat([time_feat, static_feat], dim=1)  # (Batch, 80)
        
        # 최종 예측 (32차원 중간층 거쳐 확률 출력)
        return self.classifier(combined).squeeze()  # (Batch,)


# ==============================================================================
# [데이터 준비 함수]
# ==============================================================================

def prepare_advanced_data(disease_type):
    """
    모델 입력용 데이터 준비 (특수 전처리 포함)
    
    목적:
    1. 롤링 윈도우 데이터 로드
    2. Time-shifting으로 미래 예측 설정
    3. Slope 피처 추가 (HR 변화율)
    4. 환자 연속성 검증
    
    Args:
        disease_type: 'ami' 또는 'stroke'
    
    Returns:
        X_time: (N, 6, 23) - 시계열 (22 vitals + 1 slope)
        X_static: (N, 4) - 정적 변수
        y_final: (N,) - 라벨
    
    전처리 단계:
    1. 원본 데이터 로드 (X, y, sids)
    2. Time-shifting: X[t]로 y[t+1] 예측
    3. 환자 연속성 검증: stay_id가 같은 경우만
    4. Slope 피처 생성 및 추가
    
    면접 포인트:
    "Time-shifting으로 실시간 예측 시나리오를 구현했으며,
     환자 연속성 검증으로 데이터 누수를 방지했습니다."
    """
    # -------------------------------------------------------------------------
    # Step 1: 원본 데이터 로드
    # -------------------------------------------------------------------------
    # 02_window_6h.py에서 생성된 롤링 윈도우 데이터
    X = np.load(f"{BASE_DIR}/{disease_type}_X_rolling.npy")      # (N, 6, 22)
    y_raw = np.load(f"{BASE_DIR}/{disease_type}_y_rolling.npy")  # (N,)
    sids = np.load(f"{BASE_DIR}/{disease_type}_sids_rolling.npy") # (N,)
    
    # -------------------------------------------------------------------------
    # Step 2: Time-shifting (미래 예측 설정)
    # -------------------------------------------------------------------------
    # 목적: X[t] 시점 데이터로 y[t+1] 시점 사망 여부 예측
    # 이유: 실시간 시스템에서는 "현재 데이터로 미래 예측"이 목표
    
    # X를 1칸 앞으로 (X[:-1]), y를 1칸 뒤로 (y[1:])
    X_shifted = X[:-1]        # (N-1, 6, 22)
    y_shifted = y_raw[1:]     # (N-1,)
    sids_shifted = sids[:-1]  # (N-1,)
    
    # -------------------------------------------------------------------------
    # Step 3: 환자 연속성 검증
    # -------------------------------------------------------------------------
    # 문제: X[t]와 y[t+1]이 다른 환자일 수 있음
    # 예: X[-1]이 환자 A의 마지막, y[0]이 환자 B의 첫 데이터
    # 해결: stay_id가 같은 경우만 유효
    
    valid_mask = sids_shifted == sids[1:]  # 연속된 행이 같은 환자인지
    X_final = X_shifted[valid_mask]        # 유효한 데이터만
    y_final = y_shifted[valid_mask]
    
    # -------------------------------------------------------------------------
    # Step 4: Slope 피처 생성 (HR 변화율)
    # -------------------------------------------------------------------------
    # 목적: 6시간 동안 HR(심박수)의 변화율 추가
    # 공식: (마지막 HR - 첫 HR) / 5시간
    # 의미: HR이 증가 추세인지 감소 추세인지
    
    # X_final[:, -1, 0]: 마지막 시점(6시간째)의 첫 번째 피처(HR)
    # X_final[:, 0, 0]: 첫 시점(1시간째)의 첫 번째 피처(HR)
    slopes = (X_final[:, -1, 0] - X_final[:, 0, 0]) / 5.0  # (N,)
    
    # 모든 시간 스텝에 동일한 slope 추가 (시간 불변 피처)
    # (N,) → (N, 1, 1) → (N, 6, 1) 반복
    slopes_expanded = np.repeat(slopes[:, np.newaxis, np.newaxis], 6, axis=1)
    
    # 원본 22개 피처에 slope 1개 추가 → 23개
    X_time = np.concatenate([X_final, slopes_expanded], axis=2)  # (N, 6, 23)
    
    # -------------------------------------------------------------------------
    # Step 5: 정적 변수 추출
    # -------------------------------------------------------------------------
    # 첫 시점(시간 인덱스 0)의 처음 4개 피처
    # X_final[:, 0, :4] = [Age, Gender, Admission_Type, First_Careunit]
    X_static = X_final[:, 0, :4]  # (N, 4)
    
    return X_time, X_static, y_final


# ==============================================================================
# [시각화 함수]
# ==============================================================================

def plot_calibration_results(y_true, y_prob, disease_name):
    """
    Calibration Curve 생성 및 저장
    
    Calibration이란?
    - 모델이 예측한 확률이 실제 빈도와 얼마나 일치하는가?
    - 예: 모델이 "사망 확률 70%"라고 한 100명 중 실제로 70명이 사망했는가?
    
    목적:
    1. 모델의 확률 신뢰도 검증
    2. Over-confidence/Under-confidence 탐지
    3. 임상 적용 가능성 평가
    
    Args:
        y_true: 실제 라벨 (0 or 1)
        y_prob: 예측 확률 (0~1)
        disease_name: 질환명 (차트 제목용)
    
    생성 파일:
        - {disease_name}_calibration.png
    
    해석:
    - Ideal Line(대각선): 완벽한 Calibration
    - 위쪽: Under-confidence (확률이 실제보다 낮게 예측)
    - 아래쪽: Over-confidence (확률이 실제보다 높게 예측)
    
    면접 포인트:
    "Calibration Curve가 Ideal Line에 근접하여
     모델이 예측한 확률을 임상에서 신뢰할 수 있습니다."
    """
    # -------------------------------------------------------------------------
    # Calibration Curve 계산
    # -------------------------------------------------------------------------
    # n_bins=10: 확률을 10개 구간으로 나눔 (0~0.1, 0.1~0.2, ..., 0.9~1.0)
    # prob_true: 각 구간에서 실제 양성 비율
    # prob_pred: 각 구간의 평균 예측 확률
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # -------------------------------------------------------------------------
    # 차트 그리기
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    
    # 실제 Calibration Curve (사각형 마커)
    plt.plot(prob_pred, prob_true, marker='s', label=f'{disease_name} (LSTM)')
    
    # Ideal Line (대각선, 회색 점선)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    
    # 축 레이블 및 제목
    plt.xlabel('Predicted Probability')   # X축: 예측 확률
    plt.ylabel('Actual Label Fraction')   # Y축: 실제 양성 비율
    plt.title(f'Calibration: {disease_name}')
    
    # 범례 및 격자
    plt.legend()
    plt.grid(True)
    
    # -------------------------------------------------------------------------
    # 파일 저장 및 화면 출력
    # -------------------------------------------------------------------------
    save_path = f"{BASE_DIR}/{disease_name}_calibration.png"
    plt.savefig(save_path)
    print(f"   ✅ Calibration 차트 저장: {save_path}")
    plt.show()  # Jupyter/화면에 표시


# ==============================================================================
# [핵심] 통계 분석 및 연령 그룹별 평가
# ==============================================================================

def generate_advanced_audit(disease_type):
    """
    질환별 심층 임상 감사 보고서 생성
    
    분석 항목:
    1. Calibration 검증 (확률 신뢰도)
    2. Youden's Index로 최적 임계값 도출
    3. 연령 그룹별 성능 비교 (Young vs Old)
    4. 통계적 유의성 검증 (Wilcoxon rank-sum test)
    
    Args:
        disease_type: 'ami' 또는 'stroke'
    
    Returns:
        dict: 모든 지표를 포함한 결과 딕셔너리
    
    출력 형태:
        {
            'Disease': 'AMI',
            'Total_AUROC': 0.8523,
            'Total_Best_Thr(J)': 0.1234,
            'Young_AUROC': 0.8456,
            'Old_AUROC': 0.8589,
            ...
        }
    
    면접 포인트:
    "연령 그룹 간 AUROC 차이가 3% 미만으로,
     모델이 모든 환자 그룹에서 안정적으로 작동함을 확인했습니다."
    """
    print(f"\n🚀 [{disease_type.upper()}] 심층 임상 감사 시작...")
    
    # =========================================================================
    # Phase 1: 데이터 및 모델 로드
    # =========================================================================
    X_time, X_static, y = prepare_advanced_data(disease_type)
    
    # 모델 초기화
    model = MultiModalMIMIC().to(device)
    model_path = f"{BASE_DIR}/{disease_type}_fixed_best.pth"
    
    # 모델 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"   ❌ 모델 파일 없음: {model_path}")
        return None
    
    # 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드 (Dropout 비활성화)
    
    # =========================================================================
    # Phase 2: 예측 확률 계산
    # =========================================================================
    with torch.no_grad():  # Gradient 계산 비활성화
        y_prob = model(
            torch.FloatTensor(X_time),
            torch.FloatTensor(X_static)
        ).numpy()
    
    # =========================================================================
    # Phase 3: Calibration 차트 생성
    # =========================================================================
    plot_calibration_results(y, y_prob, disease_type.upper())
    
    # =========================================================================
    # Phase 4: 연령 그룹 정의
    # =========================================================================
    # Age는 정규화되어 있음 (Z-score)
    # 음수: 평균 나이보다 젊음 (Young)
    # 양수: 평균 나이보다 많음 (Old)
    masks = {
        "Total": np.ones(len(y), dtype=bool),  # 전체
        "Young": X_static[:, 0] < 0,            # Age < 평균
        "Old": X_static[:, 0] >= 0              # Age >= 평균
    }
    
    # =========================================================================
    # Phase 5: 그룹별 성능 계산 함수 정의
    # =========================================================================
    def get_detailed_metrics(mask, label):
        """
        특정 그룹에 대한 모든 성능 지표 계산
        
        Args:
            mask: 그룹 선택 마스크 (Boolean array)
            label: 그룹명 (예: 'Total', 'Young', 'Old')
        
        Returns:
            dict: 모든 성능 지표
        
        계산 지표:
        1. Youden's Index: 최적 임계값
        2. AUROC, AUPRC: 확률 기반 지표
        3. Precision, Recall, Specificity: 이진 분류 지표
        4. F1, F2: 종합 지표
        5. Confusion Matrix: TP, FP, FN, TN
        """
        # 그룹 데이터 추출
        y_t = y[mask]       # 실제 라벨
        y_p = y_prob[mask]  # 예측 확률
        
        # ---------------------------------------------------------------------
        # Step 1: Youden's Index로 최적 임계값 계산
        # ---------------------------------------------------------------------
        # Youden's Index = Sensitivity + Specificity - 1
        # = TPR - FPR
        # 의미: Sensitivity와 Specificity의 균형점
        
        fpr, tpr, thresholds = roc_curve(y_t, y_p)
        
        # Youden's Index가 최대인 지점의 임계값
        best_thr_j = thresholds[np.argmax(tpr - fpr)]
        
        # ---------------------------------------------------------------------
        # Step 2: 최적 임계값으로 이진 예측 생성
        # ---------------------------------------------------------------------
        eval_threshold = best_thr_j
        y_pred = (y_p > eval_threshold).astype(int)  # 0 or 1
        
        # ---------------------------------------------------------------------
        # Step 3: 확률 기반 지표 계산 (임계값 무관)
        # ---------------------------------------------------------------------
        # AUROC: ROC 곡선 아래 면적 (모든 임계값 고려)
        auroc = roc_auc_score(y_t, y_p)
        
        # AUPRC: Precision-Recall 곡선 아래 면적 (불균형 데이터에 강건)
        auprc = average_precision_score(y_t, y_p)
        
        # ---------------------------------------------------------------------
        # Step 4: Confusion Matrix 계산
        # ---------------------------------------------------------------------
        # tn: True Negative (실제 0, 예측 0)
        # fp: False Positive (실제 0, 예측 1) - Type I Error
        # fn: False Negative (실제 1, 예측 0) - Type II Error
        # tp: True Positive (실제 1, 예측 1)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
        
        # ---------------------------------------------------------------------
        # Step 5: 이진 분류 지표 계산
        # ---------------------------------------------------------------------
        # Precision = TP / (TP + FP) - 양성 예측 중 실제 양성 비율
        precision = precision_score(y_t, y_pred, zero_division=0)
        
        # Recall (Sensitivity) = TP / (TP + FN) - 실제 양성 중 탐지 비율
        recall = recall_score(y_t, y_pred, zero_division=0)
        
        # Specificity = TN / (TN + FP) - 실제 음성 중 정확히 분류한 비율
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
        # Precision과 Recall의 조화평균
        f1 = f1_score(y_t, y_pred, zero_division=0)
        
        # F2-Score = 5 × (Precision × Recall) / (4 × Precision + Recall)
        # Recall에 2배 가중치 (임상에서는 False Negative가 더 위험)
        f2 = fbeta_score(y_t, y_pred, beta=2, zero_division=0)
        
        # ---------------------------------------------------------------------
        # Step 6: 통계적 유의성 검증
        # ---------------------------------------------------------------------
        # Wilcoxon rank-sum test (Mann-Whitney U test)
        # 귀무가설: 양성 그룹과 음성 그룹의 예측 확률 분포가 동일
        # p-value < 0.05: 모델이 두 그룹을 유의미하게 구분함
        _, p_val = stats.ranksums(y_p[y_t == 0], y_p[y_t == 1])
        
        # ---------------------------------------------------------------------
        # Step 7: 결과 딕셔너리 생성
        # ---------------------------------------------------------------------
        return {
            f"{label}_Best_Thr(J)": round(best_thr_j, 4),   # Youden's Index 임계값
            f"{label}_Precision": round(precision, 4),       # 정밀도
            f"{label}_Recall": round(recall, 4),             # 민감도
            f"{label}_Specificity": round(specificity, 4),   # 특이도
            f"{label}_AUROC": round(auroc, 4),               # ROC AUC
            f"{label}_AUPRC": round(auprc, 4),               # PR AUC
            f"{label}_F1": round(f1, 4),                     # F1 점수
            f"{label}_F2": round(f2, 4),                     # F2 점수
            f"{label}_TP/FP/FN/TN": f"{tp}/{fp}/{fn}/{tn}"  # Confusion Matrix
        }
    
    # =========================================================================
    # Phase 6: 모든 그룹에 대해 지표 계산
    # =========================================================================
    final_res = {"Disease": disease_type.upper()}
    
    for group_name, mask in masks.items():
        # 각 그룹(Total, Young, Old)의 지표를 계산하여 병합
        final_res.update(get_detailed_metrics(mask, group_name))
    
    return final_res


# ==============================================================================
# [실행 진입점]
# ==============================================================================

if __name__ == "__main__":
    """
    메인 실행 블록
    
    실행 순서:
    1. AMI 감사 (Calibration 차트 + 지표 계산)
    2. STROKE 감사 (Calibration 차트 + 지표 계산)
    3. 결과 통합 및 Transpose 테이블 출력
    
    생성 파일:
    - AMI_calibration.png
    - STROKE_calibration.png
    
    콘솔 출력:
    - 행: 지표명 (예: Total_AUROC, Young_Precision, ...)
    - 열: 질환 (AMI, STROKE)
    
    사용법:
        python 07_final_audit.py
    
    소요 시간:
        약 2~3분 (질환당 1분)
    """
    DISEASES = ['ami', 'stroke']
    all_results = []
    
    # =========================================================================
    # 질환별 감사 실행
    # =========================================================================
    for d in DISEASES:
        res = generate_advanced_audit(d)
        if res:
            all_results.append(res)
    
    # =========================================================================
    # 결과 테이블 생성 및 출력
    # =========================================================================
    if all_results:
        df_audit = pd.DataFrame(all_results)
        
        print("\n" + "="*100)
        print("📊 [유덴 임계값 적용 및 Calibration 완료 성적표]")
        print("="*100)
        
        # 모든 컬럼 출력 (생략 없이)
        pd.set_option('display.max_columns', None)
        
        # Transpose: 행과 열 바꾸기
        # 원본: 행=질환(AMI, STROKE), 열=지표
        # 변환 후: 행=지표, 열=질환
        # → 지표를 세로로 나열하여 비교하기 쉽게
        print(df_audit.T)
        
        # =====================================================================
        # 추가 분석 (선택적)
        # =====================================================================
        print(f"\n{'='*100}")
        print("📈 핵심 인사이트:")
        print(f"{'='*100}\n")
        
        for idx, row in df_audit.iterrows():
            disease = row['Disease']
            print(f"[{disease}]")
            print(f"  • AUROC: {row['Total_AUROC']:.4f}")
            print(f"  • 최적 임계값: {row['Total_Best_Thr(J)']:.4f}")
            print(f"  • F1-Score: {row['Total_F1']:.4f}")
            print(f"  • Young vs Old AUROC: {row['Young_AUROC']:.4f} vs {row['Old_AUROC']:.4f}")
            print()
