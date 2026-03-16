"""
03_multimodal.py (Multi-Modal LSTM 학습 및 평가)
Focal Loss(AMI) / Ensemble(STROKE) 전략으로 메인 모델을 학습하고,
Test 데이터에서 성능을 평가하며, 최적 임계값과 가중치를 확정합니다.

목적: 이미 학습된 모델을 로드하여 성능 평가 및 임계값 최적화
핵심: 
1. AMI - Focal Loss 단독 전략
2. STROKE - Ensemble (Focal 70% + Weighted 30%) 전략
3. 임계값별 성능 가이드 생성


"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import json


# ============================================================
# [환경 설정]
# ============================================================

BASE_DIR = "/home/kido/miniproject/team3"

# GPU 사용 가능 여부 확인 및 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# [모델 아키텍처 정의]
# ============================================================

class MultiModalMIMIC_Focal(nn.Module):
    """
    Focal Loss로 학습된 Multi-Modal LSTM 모델
    
    구조:
    1. LSTM: 시계열 데이터(6시간 × 22피처) 처리
    2. Static FC: 정적 데이터(Age, Gender, 가속도 46개) 처리
    3. Classifier: 두 정보를 결합하여 사망 확률 예측
    
    특징:
    - Focal Loss: 소수 클래스(사망)에 집중
    - Dropout: 과적합 방지
    - Sigmoid: 확률값 출력 (0~1)
    """
    def __init__(self, time_dim, static_dim, hidden_dim=64):
        """
        Args:
            time_dim (int): 시계열 피처 수 (22개: 11 vitals + 11 masks)
            static_dim (int): 정적 피처 수 (48개: Age, Gender, 가속도 46개)
            hidden_dim (int): LSTM 은닉층 크기 (기본 64)
        """
        super().__init__()
        
        # LSTM: 시계열 패턴 학습
        # num_layers=2: 2층 구조로 복잡한 패턴 포착
        # dropout=0.2: 레이어 간 20% 뉴런 무작위 제거 (과적합 방지)
        self.lstm = nn.LSTM(
            time_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True,  # 입력 형태: (batch, time, features)
            dropout=0.2
        )
        
        # Static FC: 정적 변수 인코딩
        # 48차원 → 32차원으로 압축
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),           # 비선형 활성화
            nn.Dropout(0.2)      # 과적합 방지
        )
        
        # Classifier: 최종 예측
        # LSTM(64) + Static(32) = 96차원 → 1차원(확률)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 32, 1),
            nn.Sigmoid()         # 0~1 확률로 변환
        )

    def forward(self, x_time, x_static):
        """
        순전파 함수
        
        Args:
            x_time: 시계열 입력 (batch, 6, 22)
            x_static: 정적 입력 (batch, 48)
        
        Returns:
            확률값 (batch,) - 24시간 내 사망 확률
        """
        # LSTM 처리: 마지막 타임스텝의 은닉 상태 사용
        # h_n: (num_layers, batch, hidden_dim)
        _, (h_n, _) = self.lstm(x_time)
        time_feat = h_n[-1]  # 마지막 레이어의 은닉 상태 (batch, 64)
        
        # Static 처리
        static_feat = self.static_fc(x_static)  # (batch, 32)
        
        # 두 정보 결합
        combined = torch.cat([time_feat, static_feat], dim=1)  # (batch, 96)
        
        # 최종 예측
        return self.classifier(combined).squeeze()  # (batch,)


class MultiModalMIMIC_Weighted(nn.Module):
    """
    Weighted BCE로 학습된 Multi-Modal LSTM 모델
    
    Focal 모델과의 차이:
    - Classifier에 Sigmoid 없음 (학습 시 BCEWithLogitsLoss 사용)
    - 추론 시 수동으로 Sigmoid 적용 필요
    """
    def __init__(self, time_dim, static_dim, hidden_dim=64):
        super().__init__()
        # 구조는 Focal과 동일
        self.lstm = nn.LSTM(time_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.static_fc = nn.Sequential(nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(0.2))
        
        # 차이점: Sigmoid 없음 (Logits 출력)
        self.classifier = nn.Linear(hidden_dim + 32, 1)

    def forward(self, x_time, x_static):
        """Logits 출력 (Sigmoid 적용 전)"""
        _, (h_n, _) = self.lstm(x_time)
        time_feat = h_n[-1] 
        static_feat = self.static_fc(x_static)
        combined = torch.cat([time_feat, static_feat], dim=1)
        return self.classifier(combined).squeeze()


# ============================================================
# [데이터 준비 함수]
# ============================================================

def prepare_clinical_data_advanced(disease_type, BASE_DIR):
    """
    모델 입력용 데이터 준비: 가속도 피처 생성
    
    과정:
    1. 롤링 윈도우 데이터 로드 (6시간 × 22피처)
    2. 가속도 피처 46개 생성 (1차 기울기 22개 + 2차 가속도 24개)
    3. 시계열(X_time)과 정적(X_static) 분리
    
    Args:
        disease_type (str): 'ami' 또는 'stroke'
        BASE_DIR (str): 데이터 경로
    
    Returns:
        X_time: 시계열 데이터 (N, 6, 22)
        X_static: 정적 데이터 (N, 48) = Age(1) + Gender(1) + 가속도(46)
        y: 라벨 (N,)
        sids: 환자 ID (N,)
    """
    # -----------------------------------------------------------
    # Step 1: 기본 데이터 로드
    # -----------------------------------------------------------
    X = np.load(f"{BASE_DIR}/{disease_type}_X_rolling.npy")    # (N, 6, 22)
    y = np.load(f"{BASE_DIR}/{disease_type}_y_rolling.npy")    # (N,)
    sids = np.load(f"{BASE_DIR}/{disease_type}_sids_rolling.npy")  # (N,)
    
    # -----------------------------------------------------------
    # Step 2: 가속도 피처 생성
    # -----------------------------------------------------------
    # X의 첫 4개 피처: Age, Gender, Admission_Type, First_Careunit
    # 나머지: 11개 vitals + 11개 masks = 22개
    n_features = X.shape[2]
    vitals = X[:, :, :n_features]  # 모든 피처 사용 (실제로는 22개)
    
    # 2-1. 1차 기울기 (Slope) 계산
    # 정의: (t=6 값 - t=0 값) / 5시간
    # 의미: 6시간 동안의 평균 변화율
    # 예: 심박수 60 → 90이면 slope = (90-60)/5 = 6 bpm/h
    slope_1st = (vitals[:, -1, :] - vitals[:, 0, :]) / 5.0  # (N, 22)
    
    # 2-2. 2차 가속도 (Acceleration) 계산
    # 정의: (마지막 구간 기울기) - (이전 구간 기울기)
    # 의미: 변화율의 변화 (가속/감속)
    # 예: 심박수가 점점 빠르게 증가 → 양의 가속도
    slope_2nd = (vitals[:, -1, :] - vitals[:, -2, :]) - (vitals[:, -2, :] - vitals[:, -3, :])  # (N, 22)
    
    # 2-3. 가속도 피처 결합
    # 총 46개: 1차 기울기 22개 + 2차 가속도 24개 (일부 중복)
    accel_features = np.concatenate([slope_1st, slope_2nd], axis=1)  # (N, 44)
    
    # 2-4. NaN/Inf 처리 (결측 또는 0으로 나눈 경우)
    # nan, inf → 0으로 대체 (보수적 접근)
    accel_features = np.nan_to_num(accel_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # -----------------------------------------------------------
    # Step 3: 시계열(X_time)과 정적(X_static) 분리
    # -----------------------------------------------------------
    # X_time: 6시간 윈도우 전체 (6, 22)
    X_time = X.astype(np.float32)
    
    # X_static: 첫 시점의 정적 변수 + 가속도 피처
    # X[:, 0, :4]: 첫 시점의 Age, Gender, Admission_Type, First_Careunit
    # accel_features: 가속도 46개
    X_static = np.concatenate([X[:, 0, :4], accel_features], axis=1).astype(np.float32)
    # 최종 형태: (N, 4+44) = (N, 48)
    
    return X_time, X_static, y, sids


# ============================================================
# [임계값 분석 함수]
# ============================================================

def generate_threshold_table(y_test, test_probs, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    """
    다양한 임계값에서의 성능 지표 계산
    
    목적:
    - 임상 현장에서 임계값 조정 시 참고용
    - Precision-Recall Trade-off 시각화
    
    예시 활용:
    - 높은 Precision 필요 시 → 높은 임계값 (0.7~0.8)
    - 높은 Recall 필요 시 → 낮은 임계값 (0.1~0.3)
    
    Args:
        y_test: 실제 라벨 (N,)
        test_probs: 예측 확률 (N,)
        thresholds: 테스트할 임계값 리스트
    
    Returns:
        pd.DataFrame: 임계값별 성능 지표 테이블
    """
    results = []
    
    for th in thresholds:
        # 현재 임계값으로 예측
        y_pred = (test_probs >= th).astype(int)
        
        # Confusion Matrix 계산
        # tn: True Negative (맞게 생존 예측)
        # fp: False Positive (잘못 사망 예측)
        # fn: False Negative (놓친 사망)
        # tp: True Positive (맞게 사망 예측)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # -----------------------------------------------------------
        # 성능 지표 계산
        # -----------------------------------------------------------
        
        # Precision (정밀도): 사망 예측 중 실제 사망 비율
        # "고위험이라고 했을 때 실제로 위험할 확률"
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall (재현율): 실제 사망 중 포착한 비율
        # "실제 사망자 중 몇 명을 찾아냈는가"
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F2-Score: Recall에 2배 가중치
        # F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        # beta=2: Recall이 Precision보다 2배 중요
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity (특이도): 실제 생존 중 맞게 예측한 비율
        # "생존자를 사망으로 오판하지 않는 능력"
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            'Threshold': th,
            'Precision': precision,
            'Recall': recall,
            'F2-Score': f2,
            'Specificity': specificity,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn)
        })
    
    return pd.DataFrame(results)


# ============================================================
# [메인 함수] 최종 모델 평가 및 확정
# ============================================================

def finalize_models():
    """
    이미 학습된 모델을 로드하여 최종 평가 및 결과 저장
    
    전략:
    - AMI: Focal Loss 단독 (Precision 우선)
    - STROKE: Ensemble (Focal 70% + Weighted 30%)
    
    출력:
    - 임계값별 성능 테이블 (CSV)
    - 예측 확률 (NPY)
    - 최종 결과 요약 (JSON)
    """
    
    # 질환별 최종 결과를 저장할 딕셔너리
    final_results = {}
    
    # -----------------------------------------------------------
    # 질환별 반복 (AMI, STROKE)
    # -----------------------------------------------------------
    for disease_type in ['ami', 'stroke']:
        print(f"\n{'='*80}")
        print(f"📊 [{disease_type.upper()}] 최종 모델 평가 및 저장")
        print(f"{'='*80}")
        
        # -------------------------------------------------------
        # Step 1: 데이터 준비
        # -------------------------------------------------------
        # 가속도 피처 포함한 데이터 로드
        X_time, X_static, y, sids = prepare_clinical_data_advanced(disease_type, BASE_DIR)
        
        # -------------------------------------------------------
        # Step 2: Train/Val/Test 분할 (환자 단위)
        # -------------------------------------------------------
        # 중요: 같은 환자가 여러 세트에 걸치지 않도록
        u_sids = np.unique(sids)
        
        # Test 20% 분리
        tr_va_sids, te_sids = train_test_split(u_sids, test_size=0.2, random_state=42)
        
        # Test 세트 마스크
        test_mask = np.isin(sids, te_sids)
        y_test = y[test_mask]
        
        # 모델 입력 차원 계산
        time_dim = X_time.shape[2]    # 22 (시계열 피처 수)
        static_dim = X_static.shape[1]  # 48 (정적 피처 수)
        
        # -------------------------------------------------------
        # Step 3: 질환별 모델 로드 및 예측
        # -------------------------------------------------------
        
        if disease_type == 'ami':
            # ---------------------------------------------------
            # AMI: Focal Loss 단독 전략
            # ---------------------------------------------------
            print("\n🎯 최종 전략: Focal Loss (Gamma=3.0) 단독 사용")
            print("   근거: Precision 0.1923으로 유일하게 임상 요구사항 만족")
            print("   설명: 침습적 검사 비용을 고려하여 오탐(FP)을 최소화")
            
            # 모델 로드
            model = MultiModalMIMIC_Focal(time_dim, static_dim).to(device)
            model.load_state_dict(torch.load(f"{BASE_DIR}/{disease_type}_multimodal_best.pth"))
            model.eval()  # 평가 모드 (Dropout 비활성화)
            
            # 예측 (배치 단위로 GPU에서 처리)
            with torch.no_grad():  # 그래디언트 계산 불필요
                X_t = torch.FloatTensor(X_time[test_mask]).to(device)
                X_s = torch.FloatTensor(X_static[test_mask]).to(device)
                final_probs = model(X_t, X_s).cpu().numpy()
            
            # 최적 임계값 (이미 계산됨)
            optimal_threshold = 0.3146
            model_type = "Focal Loss (γ=3.0)"
            
        else:
            # ---------------------------------------------------
            # STROKE: Ensemble 전략
            # ---------------------------------------------------
            print("\n🎯 최종 전략: Ensemble (Focal 70% + Weighted 30%)")
            print("   근거: F2 +0.47%p, Precision +5.2% 동시 개선")
            print("   설명: 비침습적 검사 가능하여 Recall도 중요, 균형 최적화")
            
            # Focal 모델 로드
            model_focal = MultiModalMIMIC_Focal(time_dim, static_dim).to(device)
            model_focal.load_state_dict(torch.load(f"{BASE_DIR}/{disease_type}_multimodal_best.pth"))
            model_focal.eval()
            
            # Weighted 모델 로드
            model_weighted = MultiModalMIMIC_Weighted(time_dim, static_dim).to(device)
            model_weighted.load_state_dict(torch.load(f"{BASE_DIR}/{disease_type}_weighted.pth"))
            model_weighted.eval()
            
            # 두 모델 예측 결합
            with torch.no_grad():
                X_t = torch.FloatTensor(X_time[test_mask]).to(device)
                X_s = torch.FloatTensor(X_static[test_mask]).to(device)
                
                # Focal 모델 예측 (이미 Sigmoid 적용됨)
                probs_focal = model_focal(X_t, X_s).cpu().numpy()
                
                # Weighted 모델 예측 (Logits이므로 Sigmoid 수동 적용)
                logits_weighted = model_weighted(X_t, X_s)
                probs_weighted = torch.sigmoid(logits_weighted).cpu().numpy()
                
                # Ensemble: 가중 평균
                # 70% Focal + 30% Weighted (그리드 서치로 최적 비율 도출)
                final_probs = 0.7 * probs_focal + 0.3 * probs_weighted
            
            # 최적 임계값
            optimal_threshold = 0.3859
            model_type = "Ensemble (F70:W30)"
        
        # -------------------------------------------------------
        # Step 4: 최적 임계값 기준 성능 평가
        # -------------------------------------------------------
        
        # 최적 임계값으로 예측
        y_pred_optimal = (final_probs >= optimal_threshold).astype(int)
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
        
        # 성능 지표 계산
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f2 = (5 * precision * recall) / (4 * precision + recall)
        auc = roc_auc_score(y_test, final_probs)
        
        # 결과 출력
        print(f"\n📈 최적 임계값 성능 (TH={optimal_threshold:.4f})")
        print(f"   AUC-ROC: {auc:.4f}")
        print(f"   Precision: {precision:.4f} (고위험 1000명 중 실제 {int(precision*1000)}명)")
        print(f"   Recall: {recall:.4f} (전체 사망자의 {recall*100:.1f}% 포착)")
        print(f"   F2-Score: {f2:.4f}")
        print(f"   TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        # -------------------------------------------------------
        # Step 5: 임계값별 성능 테이블 생성
        # -------------------------------------------------------
        threshold_table = generate_threshold_table(y_test, final_probs)
        threshold_table.to_csv(f"{BASE_DIR}/{disease_type}_threshold_guide.csv", index=False)
        
        print(f"\n📋 임계값별 성능 가이드 (상위 5개)")
        print(threshold_table[['Threshold', 'Precision', 'Recall', 'F2-Score']].head(5).to_string(index=False))
        
        # -------------------------------------------------------
        # Step 6: 예측 확률 및 라벨 저장
        # -------------------------------------------------------
        # 나중에 시각화, 추가 분석에 활용
        np.save(f"{BASE_DIR}/{disease_type}_final_probs.npy", final_probs)
        np.save(f"{BASE_DIR}/{disease_type}_test_labels.npy", y_test)
        
        # -------------------------------------------------------
        # Step 7: 결과 요약 딕셔너리 생성
        # -------------------------------------------------------
        final_results[disease_type] = {
            'model_type': model_type,
            'optimal_threshold': float(optimal_threshold),
            'auc_roc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f2_score': float(f2),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'test_samples': int(len(y_test)),
            'positive_rate': float(y_test.mean())  # 사망률 (클래스 불균형 참고)
        }
    
    # -----------------------------------------------------------
    # Step 8: 최종 결과 JSON 저장
    # -----------------------------------------------------------
    with open(f"{BASE_DIR}/final_results_summary.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # -----------------------------------------------------------
    # Step 9: 결과 요약 출력
    # -----------------------------------------------------------
    print(f"\n{'='*80}")
    print("✅ 최종 모델 평가 완료!")
    print(f"{'='*80}")
    
    print("\n📁 생성된 파일:")
    print(f"   1. ami_final_probs.npy - AMI 예측 확률")
    print(f"   2. ami_threshold_guide.csv - AMI 임계값 가이드")
    print(f"   3. stroke_final_probs.npy - STROKE 예측 확률")
    print(f"   4. stroke_threshold_guide.csv - STROKE 임계값 가이드")
    print(f"   5. final_results_summary.json - 전체 요약")
    
    # -----------------------------------------------------------
    # Step 10: 질환 간 비교 테이블
    # -----------------------------------------------------------
    print(f"\n📊 질환별 최종 성능 비교")
    print("-" * 80)
    print(f"{'지표':<15} | {'AMI (Focal)':<20} | {'STROKE (Ensemble)':<20}")
    print("-" * 80)
    print(f"{'Model':<15} | {final_results['ami']['model_type']:<20} | {final_results['stroke']['model_type']:<20}")
    print(f"{'AUC-ROC':<15} | {final_results['ami']['auc_roc']:<20.4f} | {final_results['stroke']['auc_roc']:<20.4f}")
    print(f"{'Precision':<15} | {final_results['ami']['precision']:<20.4f} | {final_results['stroke']['precision']:<20.4f}")
    print(f"{'Recall':<15} | {final_results['ami']['recall']:<20.4f} | {final_results['stroke']['recall']:<20.4f}")
    print(f"{'F2-Score':<15} | {final_results['ami']['f2_score']:<20.4f} | {final_results['stroke']['f2_score']:<20.4f}")
    print(f"{'Threshold':<15} | {final_results['ami']['optimal_threshold']:<20.4f} | {final_results['stroke']['optimal_threshold']:<20.4f}")
    print("-" * 80)
    
    # -----------------------------------------------------------
    # Step 11: 면접 준비 요약
    # -----------------------------------------------------------
    print("\n🎓 면접 준비 요약:")
    print("   • AMI: Precision 우선 → Focal Loss 채택")
    print("     - 이유: 침습적 검사(심도자술) 비용 고려, FP 최소화")
    print("   • STROKE: 균형 최적화 → Ensemble 채택")
    print("     - 이유: 비침습적 검사 가능, Recall도 중요")
    print("   • 질환별 임상 프로토콜 차이를 모델 선택에 반영")
    print("   • 그리드 서치로 11개 가중치 조합 탐색하여 최적값 도출")
    
    print("\n🔜 다음 단계:")
    print("   1. README.md 작성 (프로젝트 문서화)")
    print("   2. PPT 작성 (발표 준비)")
    print("   3. 면접 Q&A 준비 (핵심 질문 10개)")


# ============================================================
# [실행 진입점]
# ============================================================

if __name__ == "__main__":
    finalize_models()
