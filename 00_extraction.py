"""
MIMIC-IV ICU 사망률 예측 데이터 추출 파이프라인
목적: 심근경색(AMI)와 뇌졸중(STROKE) 환자의 ICU 입실 후 24시간 내 사망 예측을 위한 데이터 추출
"""

import pandas as pd
import numpy as np
import os


# ============================================================
# [1. 환경 설정 및 경로]
# ============================================================

# MIMIC-IV 원본 데이터가 저장된 경로 (Windows Subsystem for Linux 마운트 경로)
RAW_PATH = "/mnt/c/mini_project/subject_1_Prediction_of_Mortality/mimic4" 

# 추출된 데이터를 저장할 경로
SAVE_PATH = "/home/kido/miniproject/team3/00_extract"

# 저장 경로가 없으면 자동 생성
os.makedirs(SAVE_PATH, exist_ok=True)


# ============================================================
# [2. 분석 기준 설정]
# ============================================================

# 질환별 ICD 코드 정의
# ICD-10: 2015년 이후 사용되는 국제질병분류코드
# ICD-9: 2015년 이전 사용된 코드
ICD_CONFIG = {
    'ami': {
        'icd10': ['I21', 'I22', 'I23'],  # I21: 급성심근경색, I22: 후속심근경색, I23: 합병증
        'icd9': ['410']  # 410: 급성심근경색 (412 Old MI는 제외 - 급성기 아님)
    },
    'stroke': {
        'icd10': ['I60', 'I61', 'I63', 'I64'],  # I60: 지주막하출혈, I61: 뇌내출혈, I63: 뇌경색, I64: 뇌졸중
        'icd9': ['430', '431', '433', '434', '436']  # 430: 지주막하출혈, 431: 뇌내출혈 등
    }
}


# ============================================================
# [3. 측정 항목(ITEMID) 매핑 정의]
# ============================================================

# MIMIC-IV의 chartevents 테이블에서 사용하는 ITEMID를 임상 변수명으로 매핑
# 총 11개 동적 변수 (시간에 따라 변하는 값들)
DYNAMIC_MAPPING = {
    # -----------------------------------------------------------
    # 1. 활력징후 (Vital Signs) - 8개 피처
    # -----------------------------------------------------------
    
    # 심박수 (Heart Rate, bpm)
    220045: 'heart_rate',
    
    # 수축기 혈압 (Systolic Blood Pressure, mmHg) - 3가지 측정 방법
    220050: 'sbp',  # 비침습적 방법
    220179: 'sbp',  # 침습적 방법 1
    225309: 'sbp',  # 침습적 방법 2
    
    # 이완기 혈압 (Diastolic Blood Pressure, mmHg)
    220051: 'dbp',
    220180: 'dbp',
    225310: 'dbp',
    
    # 평균 동맥압 (Mean Arterial Pressure, mmHg) - 조직 관류 압력 지표
    220052: 'mbp',
    220181: 'mbp',
    225312: 'mbp',
    
    # 호흡수 (Respiratory Rate, breaths/min)
    220210: 'respiratory_rate',
    224690: 'respiratory_rate',
    
    # 체온 (Temperature)
    223761: 'temp_f',  # 화씨 (Fahrenheit)
    223762: 'temp_c',  # 섭씨 (Celsius)
    
    # 산소포화도 (SpO2, %) - 말초 산소 포화도
    220277: 'spo2',
    
    # 흡입산소농도 (FiO2, %) - 환자가 흡입하는 산소 비율
    223835: 'fio2',
    
    # -----------------------------------------------------------
    # 2. 의식 수준 (Glasgow Coma Scale) - 3개 피처
    # -----------------------------------------------------------
    # GCS는 의식 수준을 평가하는 지표 (3~15점, 낮을수록 위험)
    # 주의: 합산하지 않고 각각 분리하여 사용 (더 많은 정보 보존)
    
    220739: 'gcs_eye',     # 개안 반응 (1~4점)
    223900: 'gcs_verbal',  # 언어 반응 (1~5점)
    223901: 'gcs_motor',   # 운동 반응 (1~6점)

    # -----------------------------------------------------------
    # 3. 인구통계 정보 (Demographics) - 2개 피처
    # -----------------------------------------------------------
    # 주의: chartevents에 일부 기록되어 있지만, 정적 변수로 처리 예정
    
    226730: 'height',  # 신장 (cm)
    226512: 'weight'   # 체중 (kg)
}

# 추출할 ITEMID 리스트 (위 딕셔너리의 모든 키)
TARGET_ITEMIDS = list(DYNAMIC_MAPPING.keys())

# -----------------------------------------------------------
# 4. 정적 변수 (시간에 따라 변하지 않는 변수)
# -----------------------------------------------------------
STATIC_FEATURES = ['age', 'gender']

# 💡 최종 모델 입력 피처 구조:
# - 동적 피처 11개: 시계열로 6시간 윈도우에 걸쳐 측정됨
# - 정적 피처 2개: 각 환자마다 고정 값
# - 데이터프레임 상으로는 13개 컬럼이지만, 
#   모델 입력 시 시계열(6 time steps × 11 features) + 정적(2 features)로 분리


# ============================================================
# [메인 파이프라인 함수]
# ============================================================

def run_full_pipeline():
    """
    MIMIC-IV 데이터 추출의 전체 과정을 수행하는 메인 함수
    
    3단계 프로세스:
    1. 정적 코호트 확정 (대상 환자 선정)
    2. 동적 데이터 추출 (시계열 측정값)
    3. 최종 병합 및 라벨링
    """
    
    # ========================================================
    # [Step 1] 정적 마스터 데이터 로드 및 필터링
    # ========================================================
    print("🔍 [Step 1] 정적 마스터 데이터 로드 및 필터링 시작...")
    
    # -----------------------------------------------------------
    # 1.1 기본 테이블 로드
    # -----------------------------------------------------------
    
    # patients.csv: 환자 기본 정보 (성별, 나이)
    pts = pd.read_csv(
        f"{RAW_PATH}/hosp/patients.csv.gz", 
        usecols=['subject_id', 'gender', 'anchor_age']  # 필요한 컬럼만 로드 (메모리 절약)
    )
    
    # icustays.csv: ICU 입실/퇴실 정보
    icu = pd.read_csv(
        f"{RAW_PATH}/icu/icustays.csv.gz", 
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime']
        # stay_id: ICU 입실 고유 ID (분석의 기본 단위)
        # hadm_id: 병원 입원 ID (한 입원에 여러 ICU 입실 가능)
        # intime: ICU 입실 시각
        # outtime: ICU 퇴실 시각
    )
    
    # admissions.csv: 입원 정보 (사망 여부, 사망 시각)
    adm = pd.read_csv(
        f"{RAW_PATH}/hosp/admissions.csv.gz", 
        usecols=['hadm_id', 'hospital_expire_flag', 'deathtime']
        # hospital_expire_flag: 원내 사망 여부 (1=사망, 0=생존)
        # deathtime: 사망 시각 (있는 경우)
    )
    
    # diagnoses_icd.csv: 진단 코드 (질환 식별용)
    diag = pd.read_csv(
        f"{RAW_PATH}/hosp/diagnoses_icd.csv.gz", 
        dtype={'icd_code': str}  # ICD 코드를 문자열로 강제 (숫자로 읽히면 앞의 0이 사라짐)
    )

    # -----------------------------------------------------------
    # 1.2 테이블 병합 (정적 정보 통합)
    # -----------------------------------------------------------
    
    # ICU 입실 정보 + 환자 기본 정보 병합
    master = icu.merge(pts, on='subject_id', how='inner')
    
    # 입원 정보 추가 (사망 여부 포함)
    master = master.merge(adm, on='hadm_id', how='left')
    
    # -----------------------------------------------------------
    # 1.3 설계 기준 적용 (포함/제외 기준)
    # -----------------------------------------------------------
    
    # 시간 컬럼을 datetime 형식으로 변환
    master['intime'] = pd.to_datetime(master['intime'])
    master['outtime'] = pd.to_datetime(master['outtime'])
    
    # ICU 재원 기간 계산 (시간 단위)
    master['stay_duration'] = (master['outtime'] - master['intime']).dt.total_seconds() / 3600
    
    # 포함 기준 적용:
    # 1. 성인 환자 (18~89세) - anchor_age는 MIMIC-IV의 익명화된 나이
    # 2. ICU 재원 24시간 이상 - 24시간 내 사망 예측이 의미 있으려면 최소 24시간 관찰 필요
    master = master[
        (master['anchor_age'] >= 18) & 
        (master['anchor_age'] <= 89) & 
        (master['stay_duration'] >= 24)
    ]
    
    # 환자당 첫 번째 ICU 입실만 선택 (환자 독립성 확보)
    # 이유: 같은 환자의 여러 입실은 서로 독립적이지 않아 통계적 편향 발생
    master = master.sort_values(['subject_id', 'intime']).groupby('subject_id').first().reset_index()
    
    # -----------------------------------------------------------
    # 1.4 질환군별 코호트 분류
    # -----------------------------------------------------------
    
    cohorts = {}  # 질환별 데이터프레임을 저장할 딕셔너리
    total_target_stays = set()  # 전체 대상 stay_id 집합 (중복 제거용)
    
    for disease, codes in ICD_CONFIG.items():
        # ICD-10과 ICD-9 코드를 하나로 합침
        all_codes = codes['icd10'] + codes['icd9']
        
        # 진단 테이블에서 해당 질환 코드로 시작하는 입원(hadm_id) 찾기
        # startswith: 'I21' 코드는 'I210', 'I211' 등 모두 포함
        target_hadms = diag[
            diag['icd_code'].str.startswith(tuple(all_codes), na=False)
        ]['hadm_id'].unique()
        
        # 마스터 데이터에서 해당 입원에 해당하는 행만 추출
        df_disease = master[master['hadm_id'].isin(target_hadms)].copy()
        
        # 질환별 딕셔너리에 저장
        cohorts[disease] = df_disease
        
        # 전체 대상 stay_id 집합에 추가 (문자열로 변환 - 나중에 chartevents와 매칭용)
        total_target_stays.update(df_disease['stay_id'].astype(str).unique())
        
        print(f" ✅ {disease.upper()}: {len(df_disease)} stays 식별")

    # ========================================================
    # [Step 2] 동적 데이터 추출 (Chunk Processing)
    # ========================================================
    print(f"🚀 [Step 2] Chartevents 스캔 시작 (Target: {len(total_target_stays)} stays)...")
    
    # 질환별 임시 파일 경로 정의
    temp_files = {d: f"{SAVE_PATH}/{d}_temp.csv" for d in cohorts.keys()}
    
    # 기존 임시 파일 삭제 (재실행 시 중복 방지)
    for f in temp_files.values():
        if os.path.exists(f): 
            os.remove(f)

    # -----------------------------------------------------------
    # 2.1 Chartevents 청크 단위 처리
    # -----------------------------------------------------------
    # MIMIC-IV chartevents는 매우 큼 (수억 행) → 메모리 부족 방지를 위해 청크 처리 필수
    
    chunks = pd.read_csv(
        f"{RAW_PATH}/icu/chartevents.csv.gz", 
        usecols=['stay_id', 'itemid', 'charttime', 'valuenum'],  # 필요한 컬럼만
        chunksize=5000000,  # 500만 행씩 읽기
        dtype={'stay_id': str, 'itemid': int}  # 데이터 타입 지정 (메모리 최적화)
    )

    # 청크별 반복 처리
    for i, chunk in enumerate(chunks):
        # -----------------------------------------------------------
        # 2.2 1차 필터링: 대상 환자 & 대상 측정 항목만 추출
        # -----------------------------------------------------------
        filtered = chunk[
            (chunk['stay_id'].isin(total_target_stays)) &  # 우리가 관심 있는 ICU 입실만
            (chunk['itemid'].isin(TARGET_ITEMIDS))         # 우리가 정의한 11개 변수만
        ].copy()

        if not filtered.empty:
            # -----------------------------------------------------------
            # 2.3 질환군별로 분리하여 임시 파일에 저장
            # -----------------------------------------------------------
            for d, df_cohort in cohorts.items():
                # 해당 질환의 stay_id 집합
                d_stays = set(df_cohort['stay_id'].astype(str))
                
                # 현재 청크에서 해당 질환에 속하는 행만 필터링
                d_filtered = filtered[filtered['stay_id'].isin(d_stays)]
                
                if not d_filtered.empty:
                    # CSV 파일에 추가 (append mode)
                    # header는 파일이 처음 생성될 때만 작성
                    d_filtered.to_csv(
                        temp_files[d], 
                        mode='a',  # append 모드
                        index=False, 
                        header=not os.path.exists(temp_files[d])
                    )
        
        # 진행 상황 출력 (5청크마다)
        if (i+1) % 5 == 0: 
            print(f"  .. {(i+1)*5}M rows processed")

    # ========================================================
    # [Step 3] 최종 병합 및 실시간용 Offset 계산
    # ========================================================
    print("📦 [Step 3] 최종 데이터 병합 및 시계열 정렬 중...")
    
    for d, df_master in cohorts.items():
        # 임시 파일이 없으면 건너뛰기
        if not os.path.exists(temp_files[d]): 
            continue
        
        # -----------------------------------------------------------
        # 3.1 동적 데이터 로드
        # -----------------------------------------------------------
        df_dynamic = pd.read_csv(temp_files[d], dtype={'stay_id': int})
        df_dynamic['charttime'] = pd.to_datetime(df_dynamic['charttime'])
        
        # -----------------------------------------------------------
        # 3.2 마스터 정보와 병합
        # -----------------------------------------------------------
        # 각 측정값에 환자의 정적 정보 (나이, 성별, 입실 시각, 사망 정보) 추가
        final_df = df_dynamic.merge(
            df_master[['stay_id', 'intime', 'anchor_age', 'gender', 'hospital_expire_flag', 'deathtime']], 
            on='stay_id', 
            how='left'
        )
    
        # 시간 컬럼 datetime 변환
        final_df['deathtime'] = pd.to_datetime(final_df['deathtime'])
        final_df['charttime'] = pd.to_datetime(final_df['charttime'])
        
        # -----------------------------------------------------------
        # 3.3 [핵심] 동적 라벨링 로직 (24시간 내 사망 예측)
        # -----------------------------------------------------------
        
        # 측정 시점부터 사망까지 남은 시간 계산 (시간 단위)
        time_to_death = (final_df['deathtime'] - final_df['charttime']).dt.total_seconds() / 3600
        
        # 기본값: 모두 0 (생존)
        final_df['label'] = 0
        
        # 라벨 1 조건 (24시간 내 사망):
        # 1. 원내 사망 플래그가 있고 (hospital_expire_flag == 1)
        # 2. 사망 시각 정보가 있으며
        # 3. 측정 시점(charttime)으로부터 24시간 이내에 사망
        final_df.loc[
            (final_df['hospital_expire_flag'] == 1) & 
            (time_to_death > 0) &           # 측정 후 사망 (음수면 이미 사망 후 측정)
            (time_to_death <= 24),          # 24시간 이내
            'label'
        ] = 1
        
        # 💡 면접 대비:
        # - "퇴원 전 24시간 데이터가 없는 환자는?" → 라벨 0으로 처리 (보수적 접근)
        # - "측정 시점마다 라벨이 다르다" → 맞음! 시간에 따라 위험도 변화
        
        # -----------------------------------------------------------
        # 3.4 실시간 예측용 Offset 계산
        # -----------------------------------------------------------
        # ICU 입실 시각(intime) 기준으로 몇 시간 후에 측정되었는지
        # 예: offset=6 → 입실 후 6시간 시점의 측정값
        final_df['offset'] = (final_df['charttime'] - final_df['intime']).dt.total_seconds() / 3600
        
        # -----------------------------------------------------------
        # 3.5 변수명 매핑 및 단위 변환
        # -----------------------------------------------------------
        
        # ITEMID → 변수명 매핑 (예: 220045 → 'heart_rate')
        final_df['variable'] = final_df['itemid'].map(DYNAMIC_MAPPING)
        
        # 화씨 → 섭씨 변환
        f_mask = final_df['variable'] == 'temp_f'
        if f_mask.any():
            final_df.loc[f_mask, 'valuenum'] = (final_df.loc[f_mask, 'valuenum'] - 32) * 5/9
            final_df.loc[f_mask, 'variable'] = 'temp_c'  # 변수명도 temp_c로 통일
        
        # -----------------------------------------------------------
        # 3.6 컬럼명 정리 및 저장
        # -----------------------------------------------------------
        
        # anchor_age → age로 변경 (더 직관적)
        final_df = final_df.rename(columns={'anchor_age': 'age'})
        
        # 최종 저장 컬럼:
        # - stay_id: ICU 입실 ID
        # - offset: 입실 후 시간 (시간 단위)
        # - variable: 측정 변수명
        # - valuenum: 측정값
        # - age, gender: 정적 변수
        # - label: 24시간 내 사망 여부 (0/1)
        
        output_file = f"{SAVE_PATH}/{d}_raw_extracted.parquet"
        final_df[['stay_id', 'offset', 'variable', 'valuenum', 'age', 'gender', 'label']].to_parquet(
            output_file, 
            index=False
        )
        
        # 임시 파일 삭제 (디스크 공간 확보)
        os.remove(temp_files[d])
        
        print(f"✅ {d.upper()} 완료! 파일 위치: {output_file}")


# ============================================================
# [실행 진입점]
# ============================================================
if __name__ == "__main__":
    run_full_pipeline()
