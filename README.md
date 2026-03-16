# 🏥 ICU Early Mortality Risk Prediction (MIMIC-IV)

MIMIC-IV 데이터셋 기반 중환자실 조기 사망 위험도 예측 시스템. 
**AMI(급성 심근경색)**와 **Stroke(뇌졸중)** 환자를 대상으로 입실 후 48시간 내 사망 확률을 예측합니다.

---

## 🎯 주요 적용점

**1. 멀티모달 시계열 분석**  
6시간 윈도우 기반 바이탈 변화 패턴 추출 + 정적 임상 변수 결합

**2. 질환별 최적화 전략**  
AMI와 Stroke의 병태생리학적 차이를 반영한 독립 모델 개발

**3. 클래스 불균형 해결**  
SMOTE, Focal Loss(γ=3,4), Class Weight 조정 등 다층 전략 적용

**4. 해석 가능성 확보**  
- SHAP 분석을 통한 개별 예측 근거 제시  
- Permutation Importance로 모델 독립적 변수 중요도 검증  
- Calibration Curve를 통한 확률 예측 신뢰도 평가

**5. 경량화 모델**  
상위 20개 변수만 사용하는 배포용 모델 - 추론 속도 85% 향상, 성능 손실 <2%

---

## 📊 데이터셋

### MIMIC-IV v2.2
- **출처**: [PhysioNet](https://physionet.org/content/mimiciv/2.2/)
- **규모**: Beth Israel Deaconess Medical Center ICU 환자 약 38만 건 (2008-2019)
- **접근**: PhysioNet 계정 + CITI Human Research 교육 이수 필수
- **라이선스**: PhysioNet Credentialed Health Data License 1.5.0

### 본 연구 코호트
- **포함 기준**: ICU 첫 입실 환자 중 AMI 또는 Stroke 주진단
- **제외 기준**: 입실 후 6시간 이내 사망, 바이탈 데이터 50% 이상 결측
- **라벨**: 입실 후 48시간 내 원내 사망 여부 (binary)
- **변수**: 시계열 바이탈(HR, BP, SpO2, RR, Temp) + Lab 결과 + SOFA/APACHE + 인구학적 정보

---

## 📊 주요 성능 지표

### AMI (급성 심근경색) 모델

| 모델 | AUROC | AUPRC | F1-Score | F2-Score | Recall | Specificity |
|------|-------|-------|----------|----------|--------|-------------|
| **Ours (MultiModal)** | **0.8428** | **0.2381** | **0.1673** | **0.3129** | 0.7447 | 0.8040 |
| LightGBM | 0.8383 | 0.2343 | 0.1563 | 0.2984 | 0.7587 | 0.7822 |
| XGBoost | 0.8171 | 0.2095 | 0.1469 | 0.2810 | 0.7169 | 0.7798 |
| Logistic Regression | 0.8027 | 0.1449 | 0.1328 | 0.2604 | 0.7251 | 0.7481 |

### Stroke (뇌졸중) 모델

| 모델 | AUROC | AUPRC | F1-Score | F2-Score | Recall | Specificity |
|------|-------|-------|----------|----------|--------|-------------|
| **LightGBM** | **0.8910** | **0.2524** | **0.1258** | **0.2563** | 0.8301 | 0.7921 |
| Ours (MultiModal) | 0.8873 | 0.2422 | 0.1207 | 0.2477 | 0.8292 | 0.7821 |
| XGBoost | 0.8839 | 0.2250 | 0.1262 | 0.2562 | 0.8159 | 0.7968 |
| Logistic Regression | 0.8676 | 0.1151 | 0.1078 | 0.2261 | 0.8442 | 0.7471 |

> **RESULT**:  
> - AMI에서는 **MultiModal 모델**이 F2-Score 0.3129로 최고 성능 (민감도 74.5%)  
> - Stroke에서는 **LightGBM**이 AUROC 0.891, F2-Score 0.2563 달성 (민감도 83%)  
> - F2-Score 중심 평가 → False Negative 최소화에 초점 (중환자실 특성 반영)

---

## 📂 프로젝트 구조

```
release/
├── 00_extraction.py_excluded     # MIMIC-IV 데이터 추출 (admissions, icustays, chartevents)
├── 01_preprocess.py              # 결측치 처리, 이상치 제거, 파생변수 생성
├── 02_window_6h.py               # 시계열 윈도우 생성 (6시간 단위 집계)
├── 03_multimodal.py              # 시계열 + 정적 변수 융합 데이터셋 구축
├── 04_permutation_importance.py # 모델 독립적 변수 중요도 분석
├── 05_comprehensive_benchmark.py# LR/RF/XGB/LGBM/CatBoost 종합 비교
├── 06_train_lightweight_model.py# Top-20 변수 기반 경량 모델 훈련
├── 07_inference_speed.py         # 추론 시간 벤치마킹
├── 09_clinical_audit.py          # 임상 검증 (ROC, Calibration, DCA)
├── 09_shap_interpret.py          # SHAP 기반 예측 설명력 분석
├── 10_*.py                       # 최종 모델 변형 (SMOTE, Gamma, Multimodal)
├── 11_feature_importance.py      # Tree-based 특성 중요도 추출
├── 12_missing_data_analysis.py   # 결측 패턴 분석 및 보간 전략
│
├── results/
│   ├── model_benchmark/          # 8가지 모델 변형 성능 비교 (CSV, PNG)
│   ├── demographics_visualization/ # 코호트 기술통계 (연령/성별 분포, ICU 구성)
│   ├── procedure_analysis/       # 시술 빈도 (PCI, Thrombolysis 등)
│   ├── figures/                  # 성능 추이 차트 6종 (AMI/Stroke)
│   └── reports/                  # 최종 메트릭 및 ROC/PR Curve
│
├── ami_preprocess.parquet        # AMI 전처리 완료 데이터
└── stroke_preprocess.parquet     # Stroke 전처리 완료 데이터
```

---

## 🛠️ 재현 방법

### 1. MIMIC-IV 데이터 준비
```bash
# PhysioNet 계정 생성 및 CITI 교육 이수 후
# https://physionet.org/content/mimiciv/2.2/ 에서 다운로드
# 필요 테이블: admissions, icustays, chartevents, labevents, diagnoses_icd
```

### 2. 환경 설정
```bash
# 필수 라이브러리 설치 (`requirements.txt` 참조)
# Python 3.9+ 권장
pip install pandas numpy scikit-learn xgboost lightgbm catboost
pip install shap imbalanced-learn matplotlib seaborn plotly
```

### 3. 실행 순서
```bash
# Step 1: MIMIC-IV에서 AMI/Stroke 환자 추출
python 00_extraction.py

# Step 2: 전처리 (결측치 보간, 이상치 제거)
python 01_preprocess.py

# Step 3: 6시간 윈도우 시계열 특성 생성
python 02_window_6h.py

# Step 4: 멀티모달 데이터셋 구축
python 03_multimodal.py

# Step 5: 8가지 모델 벤치마킹
python 05_comprehensive_benchmark.py

# Step 6: 최종 모델 훈련
python 10_advanced_best.py          # AMI MultiModal
python 10_best_model_v3.py         # Stroke LightGBM

# Step 7: SHAP 해석 가능성 분석
python 09_shap_interpret.py
```

---

## 🔬 방법론 

### 모델링 전략
1. **Baseline**: Logistic Regression
2. **Tree Ensemble**: XGBoost, LightGBM, CatBoost + 5-Fold CV
3. **불균형 대응**: 
   - SMOTE 오버샘플링
   - Focal Loss (γ=3, 4)
   - Class Weight 조정
   - Threshold 최적화 (Youden Index)
4. **Calibration**: Platt Scaling 후처리

### 평가 지표 선정 이유
- **AUROC**: 전체 판별 능력 평가
- **AUPRC**: 불균형 데이터에서 양성 클래스 성능 평가
- **F2-Score**: Recall에 2배 가중치 → False Negative 최소화 (중환자실 특성)
- **Calibration**: 예측 확률의 신뢰도 검증

---

## 📈 주요 발견

### 변수 중요도 (SHAP Top 5)

**AMI (급성 심근경색)**:  
1. Mean Arterial Pressure (6h 평균) - 혈역학적 불안정성 지표
2. Heart Rate Variability - 자율신경계 기능 평가
3. Troponin I (peak) - 심근 손상 정도
4. Lactate (최종 측정치) - 조직 관류 상태


**Stroke (뇌졸중)**:  
1. Glasgow Coma Scale - 의식 수준
2. Systolic BP (입실 시) - 뇌관류압 평가
3. Glucose Level - 대사 이상 및 예후 인자
4. Age - 기저 생리적 예비능


> **임상적 해석**: AMI는 혈역학적 변수가, Stroke는 신경학적 변수가 예측에 핵심적 역할

### 모델별 특성 비교

**MultiModal (AMI 최적)**:
- 시계열 바이탈 변화 패턴 학습 능력 우수
- Recall 74.5% 달성 → 사망 위험 환자 조기 포착
- 단점: 추론 시간 상대적으로 길음 (실시간 배포 시 고려 필요)

**LightGBM (Stroke 최적)**:
- 정적 변수와 시계열 통계량의 효율적 조합
- Recall 83% → 뇌졸중 환자의 급격한 악화 감지
- 추론 속도 빠름 → 실시간 시스템 적합

---

## ⚠️ 한계점 및 개선 방향

**현실적 제약**:
- 단일 기관 데이터 (Beth Israel Deaconess) → 외부 타당도 미검증
- MIMIC-IV는 미국 데이터 → 국내 임상 환경 적용 시 재검증 필수
- 실시간 예측 시스템 부재 (배치 추론만 가능)
- 간호 기록, CT/MRI 영상 등 비정형 데이터 미활용
- F2-Score 최적화 → 특이도 희생 (False Positive 증가)

**향후 과제**:
1. **외부 검증**: eICU-CRD, MIMIC-IV-KR 등 다기관 데이터셋 적용
2. **딥러닝 고도화**: Temporal Convolutional Network → 장기 의존성 학습
3. **실시간 배포**: FastAPI + Docker 컨테이너화 → 병원 EMR 연동 가능성
4. **다변량 분석**: 시계열 이상치 탐지 + 생존분석 통합

---

## 📚 참고 문헌

1. Johnson, A. et al., "MIMIC-IV, a freely accessible electronic health record dataset", *Scientific Data* 10, 1 (2023).
2. Lundberg, S. M. & Lee, S., "A Unified Approach to Interpreting Model Predictions", *NeurIPS* (2017).
3. Lin, T. Y. et al., "Focal Loss for Dense Object Detection", *ICCV* (2017).
4. Goldberger, A. L. et al., "PhysioBank, PhysioToolkit, and PhysioNet", *Circulation* 101(23), e215-e220 (2000).
5. Harutyunyan, H. et al., "Multitask Learning and Benchmarking with Clinical Time Series Data", *Scientific Data* 6, 96 (2019).

---

## 📧 Contact

Jeong Ah Jin [gnokidoh@gmail.com]

