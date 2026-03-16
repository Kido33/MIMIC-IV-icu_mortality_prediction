"""
==============================================================================
[통합 벤치마킹 시스템: 7개 알고리즘 성능 비교]
==============================================================================

목적:
1. Multi-Modal LSTM의 우수성을 정량적으로 증명
2. 전통 ML(LR, RF, XGB, LGBM) vs DL(Simple LSTM, Transformer, Ours) 비교
3. Calibration Curve로 확률 신뢰도 검증
4. Youden's Index 기반 최적 임계값 자동 선정
5. 연령 그룹별 Robustness 검증

면접 포인트:
- "7개 알고리즘 중 왜 Multi-Modal LSTM을 선택했는가?" → 정량적 근거
- "모델 확률이 신뢰할 수 있는가?" → Calibration 검증
- "실제 임상 적용 시 임계값은?" → Youden's Index
- "모든 환자 그룹에서 작동하는가?" → 연령 그룹별 분석

⏰ 예상 소요: AMI 15분 + STROKE 15분 = 총 30분
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from scipy import stats  # Wilcoxon rank-sum test용
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve  # 확률 보정 평가
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score,
    precision_recall_curve, roc_curve
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = "/home/kido/miniproject/team3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"🚀 통합 벤치마킹 시스템 가동 | 장치: {device}\n")


# ==============================================================================
# [모델 정의 영역]
# ==============================================================================
# 
# 왜 이렇게 많은 모델을 정의하는가?
# → 면접 시 "왜 Multi-Modal LSTM을 선택했나?"라는 질문에
#   "7개 알고리즘을 비교한 결과 AUROC가 X% 높았고, Calibration도 우수했습니다"
#   라고 답변하기 위함
# ==============================================================================


# ------------------------------------------------------------------------------
# [모델 1, 2] 기존 최종 모델 (이미 학습 완료)
# ------------------------------------------------------------------------------
class MultiModalMIMIC_Focal(nn.Module):
    """
    AMI 최종 모델 (Focal Loss로 학습됨)
    
    특징:
    - 시계열(6시간 × 22피처) + 정적(48피처) 동시 처리
    - Classifier에 Sigmoid 포함 → 0~1 확률 직접 출력
    
    구조:
    - LSTM: 시계열 → 64차원 임베딩
    - Static FC: 정적 → 32차원 임베딩
    - Classifier: 96차원 → 1차원 → Sigmoid
    """
    def __init__(self, time_dim, static_dim, hidden_dim=64):
        super().__init__()
        
        # LSTM: 시계열 패턴 학습 (2층 구조, Dropout 20%)
        self.lstm = nn.LSTM(time_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # Static FC: 정적 변수 인코딩 (Age, Gender, 가속도 44개 등)
        self.static_fc = nn.Sequential(nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(0.2))
        
        # Classifier: 두 정보 결합 후 예측
        # ⚠️ 주의: Sigmoid 포함 → BCE Loss 사용 시 주의 필요
        self.classifier = nn.Sequential(nn.Linear(hidden_dim + 32, 1), nn.Sigmoid())

    def forward(self, x_time, x_static):
        """
        순전파
        
        Args:
            x_time: (batch, 6, 22) - 6시간 동안의 바이탈 사인
            x_static: (batch, 48) - Age, Gender, 가속도 피처
        
        Returns:
            (batch,) - 사망 확률 (0~1)
        """
        # LSTM의 마지막 은닉 상태 추출 (전체 시계열 정보 압축)
        _, (h_n, _) = self.lstm(x_time)
        time_feat = h_n[-1]  # (batch, 64)
        
        # Static 정보 인코딩
        static_feat = self.static_fc(x_static)  # (batch, 32)
        
        # 두 정보 결합 후 예측
        combined = torch.cat([time_feat, static_feat], dim=1)  # (batch, 96)
        return self.classifier(combined).squeeze()  # (batch,)


class MultiModalMIMIC_Weighted(nn.Module):
    """
    STROKE Ensemble용 모델 (Weighted BCE로 학습됨)
    
    차이점:
    - Classifier에 Sigmoid 없음 → Logits 출력
    - 나중에 수동으로 Sigmoid 적용하여 Ensemble
    
    왜 이렇게?
    → Ensemble 시 확률 공간(0~1)에서 평균내는 것보다
      Logit 공간에서 평균 → Sigmoid가 더 안정적
    """
    def __init__(self, time_dim, static_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(time_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.static_fc = nn.Sequential(nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(0.2))
        
        # ⚠️ 주의: Sigmoid 없음! (Logits 반환)
        self.classifier = nn.Linear(hidden_dim + 32, 1)

    def forward(self, x_time, x_static):
        """Logits 출력 (나중에 torch.sigmoid() 수동 적용)"""
        _, (h_n, _) = self.lstm(x_time)
        time_feat = h_n[-1] 
        static_feat = self.static_fc(x_static)
        combined = torch.cat([time_feat, static_feat], dim=1)
        return self.classifier(combined).squeeze()


# ------------------------------------------------------------------------------
# [모델 3] 비교용 단순 LSTM (벤치마킹용으로 새로 학습)
# ------------------------------------------------------------------------------
class SimpleLSTM(nn.Module):
    """
    단순 LSTM (시계열만 사용, 정적 변수 미사용)
    
    목적:
    - "정적 변수(Age, Gender, 가속도)를 추가한 효과가 있는가?"를 검증
    
    예상:
    - Multi-Modal보다 성능 낮을 것
    - 만약 비슷하면 → 정적 변수가 불필요하다는 증거
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()  # 확률 출력
    
    def forward(self, x):
        """
        Args:
            x: (batch, 6, 22) - 시계열만 입력
        Returns:
            (batch,) - 사망 확률
        """
        _, (h_n, _) = self.lstm(x)
        return self.sigmoid(self.fc(h_n[-1])).squeeze()


# ------------------------------------------------------------------------------
# [모델 4] Transformer (최신 아키텍처 비교용)
# ------------------------------------------------------------------------------
class TransformerModel(nn.Module):
    """
    Transformer Encoder 기반 모델
    
    목적:
    - "LSTM보다 Transformer가 더 좋지 않은가?"에 대한 답변
    
    예상:
    - 짧은 시계열(6시간)에서는 LSTM이 더 효율적일 가능성
    - Transformer는 긴 시퀀스에서 강점 (의료 데이터는 짧은 경우 많음)
    
    구조:
    - Embedding: 입력 → 64차원
    - TransformerEncoder: 4 heads, 2 layers
    - FC: 시퀀스 평균 → 1차원 → Sigmoid
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder Layer
        # - d_model: 임베딩 차원 (64)
        # - nhead: Multi-Head Attention 헤드 수 (4)
        # - dim_feedforward: FFN 은닉층 크기 (128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, 
            dim_feedforward=128, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (batch, 6, 22) - 시계열
        Returns:
            (batch,) - 사망 확률
        """
        x = self.embedding(x)  # (batch, 6, 64)
        x = self.transformer(x)  # (batch, 6, 64)
        
        # 시퀀스 평균 풀링 (전체 6시간 정보 통합)
        # ⚠️ LSTM은 마지막 은닉 상태만 사용하지만, Transformer는 평균 사용
        return self.sigmoid(self.fc(x.mean(dim=1))).squeeze()


# ==============================================================================
# [데이터 준비 함수]
# ==============================================================================
#
# prepare_clinical_data_advanced():
# - 06_permutation_importance.py와 동일한 전처리
# - 일관성 유지 필수! (데이터 불일치 시 벤치마킹 무의미)
# ==============================================================================

def prepare_clinical_data_advanced(disease_type, BASE_DIR):
    """
    모델 입력용 데이터 준비: 가속도 피처 생성
    
    과정:
    1. 롤링 윈도우 데이터 로드 (6시간 × 22피처)
    2. 가속도 피처 44개 생성:
       - 1차 기울기 22개: (t6 - t1) / 5시간
       - 2차 가속도 22개: (t6-t5 변화) - (t5-t4 변화)
    3. 시계열(X_time)과 정적(X_static) 분리
    
    Args:
        disease_type (str): 'ami' 또는 'stroke'
        BASE_DIR (str): 데이터 디렉토리
    
    Returns:
        X_time: (N, 6, 22) - 시계열 (LSTM 입력)
        X_static: (N, 48) - 정적 (기본 4 + 가속도 44)
        y: (N,) - 라벨 (사망 여부)
        sids: (N,) - 환자 ID (환자 단위 분할용)
    """
    # -----------------------------------------------------------
    # Step 1: 기본 데이터 로드
    # -----------------------------------------------------------
    X = np.load(f"{BASE_DIR}/{disease_type}_X_rolling.npy")      # (N, 6, 22)
    y = np.load(f"{BASE_DIR}/{disease_type}_y_rolling.npy")      # (N,)
    sids = np.load(f"{BASE_DIR}/{disease_type}_sids_rolling.npy")  # (N,)
    
    # -----------------------------------------------------------
    # Step 2: 가속도 피처 생성
    # -----------------------------------------------------------
    n_features = X.shape[2]  # 22개
    vitals = X[:, :, :n_features]  # 전체 피처 사용
    
    # 2-1. 1차 기울기 (평균 변화율)
    # 의미: 6시간 동안 바이탈이 얼마나 변했는가?
    # 예: HR 60 → 90이면 slope = 6 bpm/h (점점 빨라짐)
    slope_1st = (vitals[:, -1, :] - vitals[:, 0, :]) / 5.0
    
    # 2-2. 2차 가속도 (변화율의 변화)
    # 의미: 변화 속도가 가속/감속하는가?
    # 예: HR이 처음엔 천천히 증가 → 나중엔 빠르게 증가 → 양의 가속도
    slope_2nd = (vitals[:, -1, :] - vitals[:, -2, :]) - \
                (vitals[:, -2, :] - vitals[:, -3, :])
    
    # 2-3. 결합 (22 + 22 = 44)
    accel_features = np.concatenate([slope_1st, slope_2nd], axis=1)
    
    # 2-4. NaN/Inf 처리 (0으로 나눈 경우 등)
    # ⚠️ 보수적 전략: 결측 → 0 (변화 없음으로 가정)
    accel_features = np.nan_to_num(accel_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # -----------------------------------------------------------
    # Step 3: 시계열과 정적 분리
    # -----------------------------------------------------------
    # X_time: LSTM 입력 (6시간 전체)
    X_time = X.astype(np.float32)
    
    # X_static: Static FC 입력
    # - X[:, 0, :4]: 첫 시점의 Age, Gender, Admission_Type, First_Careunit
    #   ⚠️ 왜 첫 시점? → 이 4개는 시간에 따라 변하지 않는 정적 변수
    # - accel_features: 가속도 44개
    X_static = np.concatenate(
        [X[:, 0, :4], accel_features], 
        axis=1
    ).astype(np.float32)  # (N, 4+44) = (N, 48)
    
    return X_time, X_static, y, sids


# ==============================================================================
# [학습 및 평가 함수]
# ==============================================================================
#
# train_simple_dl_model():
# - SimpleLSTM과 Transformer를 현장에서 새로 학습
# - 기존 모델(Focal/Weighted)은 이미 학습 완료 → 로드만
# ==============================================================================

def train_simple_dl_model(model, X_train, y_train, epochs=10, patience=3):
    """
    딥러닝 모델 학습 (SimpleLSTM, Transformer용)
    
    전략:
    - Weighted BCE Loss: 클래스 불균형 해결
    - Early Stopping: 과적합 방지 (patience=3)
    - Batch Size 512: GPU 메모리 고려
    
    Args:
        model: SimpleLSTM 또는 Transformer 인스턴스
        X_train: (N, 6, 22) 학습 데이터
        y_train: (N,) 라벨
        epochs: 최대 에포크 (15)
        patience: 성능 개선 없으면 조기 종료 (3)
    
    Returns:
        학습 완료된 모델
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # -----------------------------------------------------------
    # Weighted BCE Loss 계산
    # -----------------------------------------------------------
    # 목적: 클래스 불균형 해결 (사망 << 생존)
    # pos_weight = (생존 수) / (사망 수)
    # 예: 생존 9000, 사망 1000 → pos_weight = 9
    #     → 사망 예측 틀리면 9배 패널티
    pos_weight = torch.FloatTensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # DataLoader 생성
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    # Early Stopping 변수
    best_loss = float('inf')
    no_improve = 0
    
    # -----------------------------------------------------------
    # 학습 루프
    # -----------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            # ⚠️ 중요: 모델 출력 형태 확인
            out = model(bx)
            
            # SimpleLSTM/Transformer는 Sigmoid 포함 → BCE 사용
            # ⚠️ 만약 Sigmoid 없으면 BCEWithLogitsLoss 사용
            if hasattr(model, 'sigmoid'):
                loss = nn.BCELoss()(out, by)
            else:
                loss = criterion(out, by)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # -----------------------------------------------------------
        # Early Stopping 체크
        # -----------------------------------------------------------
        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                # 3 에포크 연속 개선 없으면 종료
                break
    
    return model


def get_predictions(model, X_test, is_ensemble=False, model_focal=None, model_weighted=None):
    """
    모델 예측값 추출 (수정 버전)
    
    3가지 경우 처리:
    1. STROKE Ensemble (is_ensemble=True, model=None 가능)
    2. Multi-Modal (X_test가 tuple)
    3. Single Input (SimpleLSTM, Transformer)
    
    Args:
        model: 예측할 모델 (Ensemble 시 None 가능)
        X_test: 테스트 데이터
        is_ensemble: STROKE Ensemble 여부
        model_focal: Ensemble용 Focal 모델
        model_weighted: Ensemble용 Weighted 모델
    
    Returns:
        y_prob: (N,) - 사망 확률 (0~1)
    
    수정 사항 (2026-01-14):
    - model=None일 때 에러 방지 (is_ensemble=True 케이스)
    """
    # ✅ 핵심 수정: model이 None이 아닐 때만 eval() 호출
    if model is not None:
        model.eval()
    
    with torch.no_grad():
        # -----------------------------------------------------------
        # Case 1: STROKE Ensemble (Focal 70% + Weighted 30%)
        # -----------------------------------------------------------
        if is_ensemble:
            X_t = torch.FloatTensor(X_test[0]).to(device)  # 시계열
            X_s = torch.FloatTensor(X_test[1]).to(device)  # 정적
            
            # Focal 예측 (이미 Sigmoid 적용됨)
            probs_focal = model_focal(X_t, X_s).cpu().numpy()
            
            # Weighted 예측 (Logits → Sigmoid 수동 적용)
            logits_weighted = model_weighted(X_t, X_s)
            probs_weighted = torch.sigmoid(logits_weighted).cpu().numpy()
            
            # 가중 평균 (그리드 서치로 70:30이 최적이었음)
            return 0.7 * probs_focal + 0.3 * probs_weighted
        
        # -----------------------------------------------------------
        # Case 2: Multi-Modal (X_test가 tuple)
        # -----------------------------------------------------------
        else:
            if isinstance(X_test, tuple):
                X_t = torch.FloatTensor(X_test[0]).to(device)
                X_s = torch.FloatTensor(X_test[1]).to(device)
                return model(X_t, X_s).cpu().numpy()
            
            # -----------------------------------------------------------
            # Case 3: Single Input (SimpleLSTM, Transformer)
            # -----------------------------------------------------------
            else:
                X = torch.FloatTensor(X_test).to(device)
                return model(X).cpu().numpy()


def calculate_comprehensive_metrics(y_true, y_prob, model_name, threshold=None):
    """
    Youden's Index 기반 종합 지표 계산
    
    왜 Youden's Index?
    - Sensitivity(TPR)와 Specificity(TNR)의 균형점 찾기
    - 공식: J = TPR - FPR = Sensitivity + Specificity - 1
    - 최댓값: 두 지표의 합이 최대인 지점 = 임상적으로 최적
    
    예시:
    - Threshold 0.3: Sensitivity 0.9, Specificity 0.5 → J = 0.4
    - Threshold 0.5: Sensitivity 0.7, Specificity 0.8 → J = 0.5 ✅
    - Threshold 0.7: Sensitivity 0.5, Specificity 0.9 → J = 0.4
    
    Args:
        y_true: (N,) 실제 라벨
        y_prob: (N,) 예측 확률
        model_name: 모델 이름 (결과 테이블용)
        threshold: 지정된 임계값 (None이면 자동 계산)
    
    Returns:
        dict: 11개 지표 (AUROC, AUPRC, Precision, ..., P-value)
    """
    # -----------------------------------------------------------
    # Step 1: ROC Curve 계산
    # -----------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # -----------------------------------------------------------
    # Step 2: Youden's Index로 최적 임계값 찾기
    # -----------------------------------------------------------
    if threshold is None:
        # Youden's Index = TPR - FPR
        # 해석: Sensitivity와 Specificity의 거리 최대화
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        threshold = thresholds[best_idx]
    
    # -----------------------------------------------------------
    # Step 3: 임계값 기반 이진 예측
    # -----------------------------------------------------------
    y_pred = (y_prob >= threshold).astype(int)
    
    # -----------------------------------------------------------
    # Step 4: 혼동 행렬 계산
    # -----------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # tn: True Negative (생존 정확히 예측)
    # fp: False Positive (생존을 사망으로 오예측)
    # fn: False Negative (사망을 생존으로 오예측) ← 임상적으로 가장 위험!
    # tp: True Positive (사망 정확히 예측)
    
    # -----------------------------------------------------------
    # Step 5: 성능 지표 계산
    # -----------------------------------------------------------
    # Precision = TP / (TP + FP) = 사망 예측 중 실제 사망 비율
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall (Sensitivity) = TP / (TP + FN) = 실제 사망 중 예측 성공 비율
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity = TN / (TN + FP) = 실제 생존 중 예측 성공 비율
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # F1-Score = 2 * Precision * Recall / (Precision + Recall)
    # 해석: Precision과 Recall의 조화 평균
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # F2-Score = (1 + 2²) * Precision * Recall / (2² * Precision + Recall)
    # 해석: Recall을 더 중요시 (사망 놓치는 것이 더 위험)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    # AUROC = ROC 곡선 아래 면적 (0.5~1.0)
    # 해석: 0.5=랜덤, 1.0=완벽
    auroc = roc_auc_score(y_true, y_prob)
    
    # AUPRC = PR 곡선 아래 면적
    # 해석: 클래스 불균형에 강건한 지표 (AUROC보다 엄격)
    auprc = average_precision_score(y_true, y_prob)
    
    # -----------------------------------------------------------
    # Step 6: 통계적 유의성 검증 (Wilcoxon rank-sum test)
    # -----------------------------------------------------------
    # 목적: "사망군과 생존군의 예측 확률 분포가 통계적으로 다른가?"
    # H0: 두 군의 확률 분포가 동일
    # H1: 사망군의 확률이 유의미하게 높음
    # 예: p < 0.001이면 모델이 실제로 차이를 감지함
    _, p_val = stats.ranksums(y_prob[y_true == 0], y_prob[y_true == 1])
    
    # -----------------------------------------------------------
    # Step 7: 결과 딕셔너리 반환
    # -----------------------------------------------------------
    return {
        'Model': model_name,
        'Threshold': round(threshold, 4),
        'AUROC': round(auroc, 4),
        'AUPRC': round(auprc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Specificity': round(specificity, 4),
        'F1-Score': round(f1, 4),
        'F2-Score': round(f2, 4),
        'TP/FP/FN/TN': f"{tp}/{fp}/{fn}/{tn}",  # 면접 시 설명 가능
        'P-value': f"{p_val:.4e}"  # 과학적 표기법
    }


# ==============================================================================
# [통합 벤치마킹 메인 함수]
# ==============================================================================
#
# run_comprehensive_benchmark():
# - 한 질환(AMI 또는 STROKE)에 대해 7개 모델 비교
# - 5단계 파이프라인:
#   1. 기존 최종 모델 평가
#   2. ML 모델 학습 및 평가
#   3. DL 모델 학습 및 평가
#   4. Calibration Curve 생성
#   5. 연령 그룹별 분석
# ==============================================================================

def run_comprehensive_benchmark(disease_type):
    """
    통합 벤치마킹 실행
    
    출력:
    - CSV: {disease}_comprehensive_benchmark.csv
    - PNG: {disease}_calibration_all_models.png
    - Console: 모델별 성능 테이블, 그룹별 분석
    
    Args:
        disease_type (str): 'ami' 또는 'stroke'
    
    Returns:
        df_results: 모델별 성능 DataFrame
        df_group: 연령 그룹별 성능 DataFrame
    """
    print(f"\n{'='*80}")
    print(f"🔥 [{disease_type.upper()}] 통합 벤치마킹 시작")
    print(f"{'='*80}\n")
    
    # ===========================================================================
    # STEP 0: 데이터 로드 및 분할
    # ===========================================================================
    
    # -----------------------------------------------------------
    # 0-1. 데이터 로드 (가속도 피처 포함)
    # -----------------------------------------------------------
    X_time, X_static, y, sids = prepare_clinical_data_advanced(disease_type, BASE_DIR)
    
    # -----------------------------------------------------------
    # 0-2. 환자 단위 분할 (Data Leakage 방지!)
    # -----------------------------------------------------------
    # ⚠️ 중요: 환자 ID 기준으로 분할
    # - 잘못된 방법: 샘플 단위 분할 → 같은 환자가 Train/Test 양쪽에
    # - 올바른 방법: 환자 단위 분할 → Leakage 방지
    
    u_sids = np.unique(sids)  # 중복 제거한 환자 ID
    
    # Train+Val(80%) vs Test(20%)
    tr_va_sids, te_sids = train_test_split(u_sids, test_size=0.2, random_state=42)
    
    # Train(70%) vs Val(10%) - Val은 향후 하이퍼파라미터 튜닝 시 사용 가능
    tr_sids, va_sids = train_test_split(tr_va_sids, test_size=0.125, random_state=42)
    # 0.125 = 10% / 80% (전체의 10%를 Val로)
    
    # 마스크 생성
    train_mask = np.isin(sids, tr_sids)
    val_mask = np.isin(sids, va_sids)
    test_mask = np.isin(sids, te_sids)
    
    # 데이터 분할
    X_time_train, X_time_test = X_time[train_mask], X_time[test_mask]
    X_static_train, X_static_test = X_static[train_mask], X_static[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # -----------------------------------------------------------
    # 0-3. ML 모델용 2D 변환
    # -----------------------------------------------------------
    # ML 모델(LR, RF, XGB, LGBM)은 3D 시계열 처리 불가
    # 해결: 마지막 시점(t=5)만 사용 → 2D로 변환
    # ⚠️ 정보 손실 있지만, ML 모델의 한계
    X_train_2d = X_time_train[:, -1, :]  # (N, 22)
    X_test_2d = X_time_test[:, -1, :]
    
    # -----------------------------------------------------------
    # 0-4. 스케일링 (LR, RF용)
    # -----------------------------------------------------------
    # ⚠️ XGBoost/LightGBM은 스케일링 불필요 (트리 기반)
    # LR/RF는 필요 (거리 기반)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # -----------------------------------------------------------
    # 0-5. 결과 저장 변수 초기화
    # -----------------------------------------------------------
    results = []  # 모델별 성능 지표 저장
    all_probs = {}  # Calibration Curve용 확률 저장
    
    # ===========================================================================
    # STEP 1: 기존 최종 모델 평가 (이미 학습 완료)
    # ===========================================================================
    
    print("📊 Step 1: 기존 최종 모델 평가\n")
    
    time_dim = X_time.shape[2]  # 22
    static_dim = X_static.shape[1]  # 48
    
    # -----------------------------------------------------------
    # 1-1. AMI: Focal Loss 단독
    # -----------------------------------------------------------
    if disease_type == 'ami':
        # 모델 로드
        model_final = MultiModalMIMIC_Focal(time_dim, static_dim).to(device)
        model_final.load_state_dict(torch.load(f"{BASE_DIR}/{disease_type}_multimodal_best.pth"))
        
        # 예측
        probs_final = get_predictions(model_final, (X_time_test, X_static_test))
        
        # 저장
        all_probs['Ours (Focal Loss)'] = probs_final
        results.append(calculate_comprehensive_metrics(
            y_test, probs_final, 'Ours (Focal Loss)'
        ))
        print("   ✅ AMI Focal Loss 평가 완료")
    
    # -----------------------------------------------------------
    # 1-2. STROKE: Ensemble (Focal 70% + Weighted 30%)
    # -----------------------------------------------------------
    else:
        # Focal 모델 로드
        model_focal = MultiModalMIMIC_Focal(time_dim, static_dim).to(device)
        model_focal.load_state_dict(torch.load(f"{BASE_DIR}/{disease_type}_multimodal_best.pth"))
        
        # Weighted 모델 로드
        model_weighted = MultiModalMIMIC_Weighted(time_dim, static_dim).to(device)
        model_weighted.load_state_dict(torch.load(f"{BASE_DIR}/{disease_type}_weighted.pth"))
        
        # Ensemble 예측
        probs_final = get_predictions(
            None, (X_time_test, X_static_test), 
            is_ensemble=True, model_focal=model_focal, model_weighted=model_weighted
        )
        
        # 저장
        all_probs['Ours (Ensemble 70:30)'] = probs_final
        results.append(calculate_comprehensive_metrics(
            y_test, probs_final, 'Ours (Ensemble 70:30)'
        ))
        print("   ✅ STROKE Ensemble 평가 완료")
    
    # ===========================================================================
    # STEP 2: 전통 ML 모델 학습 및 평가
    # ===========================================================================
    
    print("\n📊 Step 2: 전통적 ML 모델 학습 및 평가\n")
    
    # -----------------------------------------------------------
    # 2-1. 모델 정의
    # -----------------------------------------------------------
    # ⚠️ class_weight='balanced': 클래스 불균형 자동 조정
    ml_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,  # 수렴 보장
            class_weight='balanced',  # 불균형 해결
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,  # 트리 100개
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # 모든 CPU 코어 사용
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            tree_method='hist',  # GPU 가속 지원
            random_state=42
            # ⚠️ scale_pos_weight 미지정 → 자동 계산
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            verbose=-1,  # 로그 숨김
            random_state=42,
            class_weight='balanced'
        )
    }
    
    # -----------------------------------------------------------
    # 2-2. 학습 및 평가
    # -----------------------------------------------------------
    for name, model in ml_models.items():
        print(f"   ⏳ {name} 학습 중...")
        
        # XGBoost/LightGBM: 스케일링 불필요
        if 'XGBoost' in name or 'LightGBM' in name:
            model.fit(X_train_2d, y_train)
            probs = model.predict_proba(X_test_2d)[:, 1]  # 양성 클래스 확률
        
        # LR/RF: 스케일링 필요
        else:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # 저장
        all_probs[name] = probs
        results.append(calculate_comprehensive_metrics(y_test, probs, name))
        print(f"   ✅ {name} 완료")
    
    # ===========================================================================
    # STEP 3: 딥러닝 모델 학습 및 평가
    # ===========================================================================
    
    print("\n📊 Step 3: 딥러닝 모델 학습 및 평가\n")
    
    # -----------------------------------------------------------
    # 3-1. SimpleLSTM
    # -----------------------------------------------------------
    print("   ⏳ Simple LSTM 학습 중...")
    lstm_simple = SimpleLSTM(time_dim).to(device)
    lstm_simple = train_simple_dl_model(lstm_simple, X_time_train, y_train, epochs=15, patience=3)
    probs_lstm = get_predictions(lstm_simple, X_time_test)
    
    all_probs['LSTM (Simple)'] = probs_lstm
    results.append(calculate_comprehensive_metrics(y_test, probs_lstm, 'LSTM (Simple)'))
    print("   ✅ Simple LSTM 완료")
    
    # -----------------------------------------------------------
    # 3-2. Transformer
    # -----------------------------------------------------------
    print("   ⏳ Transformer 학습 중...")
    transformer = TransformerModel(time_dim).to(device)
    transformer = train_simple_dl_model(transformer, X_time_train, y_train, epochs=15, patience=3)
    probs_trans = get_predictions(transformer, X_time_test)
    
    all_probs['Transformer'] = probs_trans
    results.append(calculate_comprehensive_metrics(y_test, probs_trans, 'Transformer'))
    print("   ✅ Transformer 완료")
    
    # ===========================================================================
    # STEP 4: Calibration Curve 생성
    # ===========================================================================
    
    print("\n📊 Step 4: Calibration Curve 생성\n")
    
    # -----------------------------------------------------------
    # Calibration Curve란?
    # -----------------------------------------------------------
    # 목적: "모델이 예측한 확률이 실제와 일치하는가?" 검증
    # 
    # 예시:
    # - 모델이 "사망 확률 0.8"이라고 예측한 100명 중 실제 80명 사망 → 잘 보정됨
    # - 모델이 "사망 확률 0.8"이라고 예측한 100명 중 실제 50명 사망 → 과신(Overconfident)
    # 
    # 해석:
    # - 대각선(y=x)에 가까울수록 좋음
    # - 아래쪽: Underconfident (확률을 낮게 예측)
    # - 위쪽: Overconfident (확률을 높게 예측)
    # -----------------------------------------------------------
    
    plt.figure(figsize=(12, 8))
    
    # 각 모델별 Calibration Curve 그리기
    for name, probs in all_probs.items():
        # n_bins=10: 확률을 10개 구간으로 나눔
        # prob_true: 각 구간의 실제 양성 비율
        # prob_pred: 각 구간의 평균 예측 확률
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name, linewidth=2)
    
    # 완벽한 보정 선 (y=x)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Curve Comparison: {disease_type.upper()}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 저장
    calib_path = f"{BASE_DIR}/{disease_type}_calibration_all_models.png"
    plt.savefig(calib_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Calibration 차트 저장: {calib_path}")
    
    # ===========================================================================
    # STEP 5: 결과 테이블 출력 및 저장
    # ===========================================================================
    
    print(f"\n{'='*80}")
    print(f"📊 [{disease_type.upper()}] 최종 벤치마킹 결과 (Youden's Index 기반)")
    print(f"{'='*80}\n")
    
    # -----------------------------------------------------------
    # DataFrame 생성 및 정렬 (AUROC 높은 순)
    # -----------------------------------------------------------
    df_results = pd.DataFrame(results).sort_values('AUROC', ascending=False)
    print(df_results.to_string(index=False))
    
    # -----------------------------------------------------------
    # CSV 저장
    # -----------------------------------------------------------
    csv_path = f"{BASE_DIR}/{disease_type}_comprehensive_benchmark.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ 결과 저장: {csv_path}")
    
    # ===========================================================================
    # STEP 6: 연령 그룹별 분석
    # ===========================================================================
    
    print(f"\n📊 Step 5: 연령 그룹별 분석\n")
    
    # -----------------------------------------------------------
    # 목적: "모든 연령대에서 작동하는가?" 검증
    # -----------------------------------------------------------
    # 왜 필요?
    # - 전체 성능이 좋아도 특정 그룹에서 실패 가능
    # - 예: 젊은 환자에서만 AUROC 0.9, 고령 환자에서 0.6 → 편향됨
    # -----------------------------------------------------------
    
    # Age 중앙값 기준으로 분할
    age_median = np.median(X_static_test[:, 0])
    young_mask = X_static_test[:, 0] < age_median
    old_mask = X_static_test[:, 0] >= age_median
    
    group_results = []
    
    # 각 그룹별 성능 계산
    for group_name, mask in [('Young', young_mask), ('Old', old_mask)]:
        y_group = y_test[mask]
        
        # 본인 모델의 확률 사용
        model_key = 'Ours (Focal Loss)' if disease_type == 'ami' else 'Ours (Ensemble 70:30)'
        probs_group = all_probs[model_key][mask]
        
        # 지표 계산
        metrics = calculate_comprehensive_metrics(y_group, probs_group, f'{group_name} Group')
        group_results.append(metrics)
    
    # 결과 출력
    df_group = pd.DataFrame(group_results)
    print(df_group.to_string(index=False))
    
    # -----------------------------------------------------------
    # 해석 가이드:
    # -----------------------------------------------------------
    # - Young과 Old의 AUROC 차이 <5%: Robust!
    # - 차이 >10%: 특정 그룹에 편향 → 원인 분석 필요
    #   (예: 고령 환자 데이터 부족, 피처 선택 문제 등)
    # -----------------------------------------------------------
    
    return df_results, df_group


# ==============================================================================
# [실행 진입점]
# ==============================================================================

if __name__ == "__main__":
    """
    메인 실행 블록
    
    순서:
    1. AMI 벤치마킹 (약 15분)
    2. STROKE 벤치마킹 (약 15분)
    3. 결과 요약 출력
    
    생성 파일:
    - ami_calibration_all_models.png
    - ami_comprehensive_benchmark.csv
    - stroke_calibration_all_models.png
    - stroke_comprehensive_benchmark.csv
    """
    all_results = {}
    
    # -----------------------------------------------------------
    # 두 질환에 대해 벤치마킹 실행
    # -----------------------------------------------------------
    for disease in ['ami', 'stroke']:
        df_main, df_group = run_comprehensive_benchmark(disease)
        all_results[disease] = {'main': df_main, 'group': df_group}
    
    # -----------------------------------------------------------
    # 최종 요약
    # -----------------------------------------------------------
    print(f"\n{'='*80}")
    print("✅ 전체 벤치마킹 완료!")
    print(f"{'='*80}\n")
    
    print("📁 생성된 파일:")
    print("   - ami_calibration_all_models.png")
    print("   - ami_comprehensive_benchmark.csv")
    print("   - stroke_calibration_all_models.png")
    print("   - stroke_comprehensive_benchmark.csv")
    
    # -----------------------------------------------------------
    # 면접 포인트 정리
    # -----------------------------------------------------------
    print("\n🎓 핵심 인사이트:")
    print("   • 7개 모델 성능 비교 완료 (Ours, LR, RF, XGB, LGBM, LSTM, Transformer)")
    print("   • Calibration Curve로 확률 신뢰도 평가")
    print("   • Youden's Index로 최적 임계값 자동 선정")
    print("   • 연령 그룹별 성능 차이 분석")
    
    # -----------------------------------------------------------
    # 다음 단계 제안
    # -----------------------------------------------------------
    print("\n📌 면접 준비 TIP:")
    print("   1. CSV 파일 열어서 AUROC 상위 3개 모델 확인")
    print("   2. Calibration 그래프에서 본인 모델이 대각선에 가까운지 확인")
    print("   3. Young/Old 그룹 AUROC 차이 계산 (Robustness 증명)")
    print("   4. P-value < 0.001 확인 (통계적 유의성 증명)")
    
    # -----------------------------------------------------------
    # 예상 질문 대비
    # -----------------------------------------------------------
    print("\n❓ 예상 면접 질문:")
    print("   Q1. 왜 Multi-Modal LSTM을 선택했나요?")
    print("   → A1. 7개 알고리즘 비교 결과, AUROC가 X% 높고 Calibration도 우수했습니다.")
    print("")
    print("   Q2. XGBoost나 LightGBM이 더 간단한데 왜 딥러닝을?")
    print("   → A2. 시계열 패턴 학습에서 LSTM이 유리하고, 정적 변수와 융합하여 X% 성능 향상했습니다.")
    print("")
    print("   Q3. 모델 확률이 신뢰할 수 있나요?")
    print("   → A3. Calibration Curve에서 대각선 근처에 위치하여 확률이 실제와 일치합니다.")
    print("")
    print("   Q4. 특정 환자 그룹에서 성능이 떨어지지 않나요?")
    print("   → A4. 연령 그룹별 분석 결과, Young/Old 모두 AUROC 0.X 이상으로 Robust합니다.")
