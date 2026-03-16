"""
MIMIC-IV ICU 데이터 전처리 파이프라인
목적: 추출된 Long Format 데이터를 모델 학습용 Wide Format으로 변환
주요 기법: 
1. 서울대 논문 기반 임상적 정상 범위 필터링
2. RealMIP 논문 기반 급격한 변화율(Delta) 제거
3. Missing 마스킹 및 LOCF 보간
"""

import pandas as pd
import numpy as np
import os
import logging


# ============================================================
# [임상적 정상 범위 정의]
# ============================================================

# 각 활력징후 및 측정값의 생리학적으로 가능한 범위
# 출처: 서울대학교 논문 및 임상 가이드라인
# 범위를 벗어나는 값은 측정 오류 또는 기기 결함으로 간주
VALID_RANGES = {
    # 활력징후 (Vital Signs)
    'heart_rate': [10, 190],        # 심박수 (bpm): 극단적 서맥/빈맥 범위
    'sbp': [40, 230],                # 수축기 혈압 (mmHg): 쇼크~고혈압 응급
    'dbp': [20, 130],                # 이완기 혈압 (mmHg)
    'mbp': [30, 150],                # 평균 동맥압 (mmHg): 조직 관류 기준
    'respiratory_rate': [5, 50],     # 호흡수 (breaths/min): 무호흡~과호흡
    'temp_c': [32, 45],              # 체온 (섭씨): 저체온~고열
    'spo2': [60, 100],               # 산소포화도 (%): 중증 저산소증~정상
    'fio2': [20, 100],               # 흡입산소농도 (%): 실내공기~순수산소
    
    # 의식 수준 (Glasgow Coma Scale)
    'gcs_eye': [1, 4],               # 개안 반응: 1(무반응)~4(자발적)
    'gcs_verbal': [1, 5],            # 언어 반응: 1(무반응)~5(정상)
    'gcs_motor': [1, 6],             # 운동 반응: 1(무반응)~6(정상)
    
    # 신체 계측 (Demographics)
    'height': [100, 250],            # 신장 (cm): 소아~거인증 범위
    'weight': [30, 250]              # 체중 (kg): 극단적 저체중~비만
}


# ============================================================
# [통합 전처리 클래스]
# ============================================================

class SNU_RealMIP_UnifiedCleaner:
    """
    서울대 + RealMIP 논문 기반 통합 전처리 클래스
    
    주요 기능:
    1. 임상적 이상치 제거 (Static Range Filtering)
    2. 급격한 변화율 제거 (Dynamic Delta Filtering)
    3. 시간 버킷팅 및 피벗 변환
    4. Missing 마스킹 (결측 여부 기록)
    5. LOCF 보간 및 정규화
    """
    
    def __init__(self, valid_ranges):
        """
        클래스 초기화
        
        Args:
            valid_ranges (dict): 변수별 임상적 정상 범위 딕셔너리
        """
        self.ranges = valid_ranges
        
        # 로깅 설정 (진행 상황 추적용)
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    # ========================================================
    # [핵심 메서드 1] 통합 이상치 필터링
    # ========================================================
    
    def _filter_clinical_outliers(self, df, threshold_pct=0.5):
        """
        임상적 이상치 제거: 2단계 필터링
        
        단계 A: 생리학적 절대 범위 컷오프 (Static)
        - 예: 심박수 300 bpm → NaN (인간에게 불가능)
        
        단계 B: 시간당 급격한 변화율 제거 (Dynamic)
        - 예: 심박수 80 → 160 (1시간 내 100% 증가) → 의심
        - RealMIP 논문: "1시간에 50% 이상 변화는 측정 오류 가능성 높음"
        
        Args:
            df (pd.DataFrame): Long Format 데이터
            threshold_pct (float): 변화율 임계값 (기본 50%)
        
        Returns:
            pd.DataFrame: 이상치가 제거된 데이터
        """
        self.logger.info("🛠️ Applying Unified Clinical & Delta Outlier Filtering...")
        
        # -----------------------------------------------------------
        # A. 생리학적 범위 컷오프 (Static Denoise)
        # -----------------------------------------------------------
        # 각 변수별로 정상 범위를 벗어나는 값을 NaN으로 대체
        
        for var_name, (low, high) in self.ranges.items():
            # 현재 변수에 해당하는 행만 선택
            mask = df['variable'] == var_name
            
            # 범위 밖 값을 NaN으로 설정
            # 예: heart_rate < 10 or heart_rate > 190 → NaN
            df.loc[mask & ((df['valuenum'] < low) | (df['valuenum'] > high)), 'valuenum'] = np.nan
        
        # -----------------------------------------------------------
        # B. 급격한 변화율 필터링 (Dynamic Delta)
        # -----------------------------------------------------------
        # RealMIP 핵심 기법: "짧은 시간에 너무 큰 변화는 의심"
        
        # 1. 시간순 정렬 (같은 환자, 같은 변수 내에서)
        df = df.sort_values(['stay_id', 'variable', 'offset'])
        
        # 2. 이전 측정값 가져오기 (같은 환자, 같은 변수의 바로 전 값)
        df['prev_val'] = df.groupby(['stay_id', 'variable'])['valuenum'].shift(1)
        
        # 3. 변화율 계산 (절대값)
        # delta_pct = |현재값 - 이전값| / 이전값
        # 예: 80 → 120이면 (120-80)/80 = 0.5 (50% 증가)
        df['delta_pct'] = (df['valuenum'] - df['prev_val']).abs() / (df['prev_val'] + 1e-8)
        # 1e-8: 0으로 나누기 방지용 작은 값
        
        # 4. GCS는 단계적 변화가 정상이므로 제외
        # 예: GCS 눈 반응 1→4는 의식 회복으로 정상
        # Vital Signs만 변화율 필터링 적용
        mask_vitals = ~df['variable'].str.contains('gcs')
        
        # 5. 변화율이 임계값(50%)을 초과하는 값을 NaN 처리
        noise_mask = mask_vitals & (df['delta_pct'] > threshold_pct)
        df.loc[noise_mask, 'valuenum'] = np.nan
        
        # 6. NaN 행 제거 및 임시 컬럼 삭제
        return df.dropna(subset=['valuenum']).drop(columns=['prev_val', 'delta_pct'])

    # ========================================================
    # [핵심 메서드 2] 전체 파이프라인
    # ========================================================
    
    def process_unified(self, disease_type, input_dir, output_dir):
        """
        전체 전처리 파이프라인 실행
        
        흐름:
        1. 원본 데이터 로드 (Long Format)
        2. 통합 이상치 필터링
        3. 시간 버킷팅 (1시간 단위 집계)
        4. Pivot 변환 (Long → Wide Format)
        5. Missing 마스킹 (결측 여부 기록)
        6. LOCF 보간 및 평균 대체
        7. Z-score 정규화
        8. 성별 인코딩
        9. 최종 저장
        
        Args:
            disease_type (str): 'ami' 또는 'stroke'
            input_dir (str): 원본 데이터 경로
            output_dir (str): 저장 경로
        
        Returns:
            pd.DataFrame: 전처리 완료된 Wide Format 데이터
        """
        # -----------------------------------------------------------
        # Step 0: 파일 로드
        # -----------------------------------------------------------
        input_file = f"{input_dir}/{disease_type}_raw_extracted.parquet"
        
        if not os.path.exists(input_file):
            self.logger.error(f"❌ 파일을 찾을 수 없습니다: {input_file}")
            return None
        
        # Parquet 포맷 로드 (CSV보다 5~10배 빠름)
        df = pd.read_parquet(input_file)
        
        # -----------------------------------------------------------
        # Step 1: 통합 이상치 제거
        # -----------------------------------------------------------
        df = self._filter_clinical_outliers(df)

        # -----------------------------------------------------------
        # Step 2: 시간 버킷팅 (1시간 단위)
        # -----------------------------------------------------------
        # offset은 소수점 (예: 6.5시간)인데, 1시간 단위로 집계
        # 예: offset 6.2, 6.5, 6.8 → 모두 time_bucket=6
        df['time_bucket'] = df['offset'].astype(int)

        # -----------------------------------------------------------
        # Step 3: Pivot 변환 (Long → Wide Format)
        # -----------------------------------------------------------
        self.logger.info(f"🔄 {disease_type.upper()} Pivoting with Metadata preservation...")
        
        # Long Format 예시:
        # stay_id | time_bucket | variable     | valuenum
        # 1001    | 6           | heart_rate   | 80
        # 1001    | 6           | sbp          | 120
        
        # Wide Format 목표:
        # stay_id | time_bucket | age | gender | label | heart_rate | sbp | ...
        # 1001    | 6           | 65  | 0      | 1     | 80         | 120 | ...
        
        # 인덱스로 사용할 컬럼 (정적 변수 + 식별자)
        target_indices = ['stay_id', 'time_bucket', 'age', 'gender', 'label']
        
        # 실제로 데이터에 존재하는 컬럼만 선택 (안전성)
        available_indices = [idx for idx in target_indices if idx in df.columns]
        
        # Pivot Table 생성
        pivot_df = df.pivot_table(
            index=available_indices,    # 행: 환자 + 시간 + 정적 변수
            columns='variable',          # 열: 측정 변수명 (heart_rate, sbp 등)
            values='valuenum',           # 값: 측정값
            aggfunc='last'              # 같은 시간대에 여러 측정값이 있으면 마지막 값 사용
        ).reset_index()
        
        # reset_index(): Multi-Index를 일반 컬럼으로 변환

        # -----------------------------------------------------------
        # Step 4: Masking & Imputation (RealMIP 핵심 로직)
        # -----------------------------------------------------------
        # 결측 처리의 핵심: "값이 없다는 정보도 중요하다"
        # 예: 혈압이 안 재어진 것 = 안정적 or 측정 불가 상태
        
        # 4-1. 피처 컬럼 식별 (측정 변수만)
        feature_cols = [c for c in pivot_df.columns if c in self.ranges.keys()]
        
        # 4-2. 시간순 정렬
        pivot_df = pivot_df.sort_values(['stay_id', 'time_bucket'])
        
        # 4-3. Mask 생성: 실제 측정값이 있었는지 기록
        # 예: heart_rate가 NaN이면 mask_heart_rate=0, 값이 있으면 1
        mask_df = pivot_df[feature_cols].notna().astype(np.float32)
        mask_df.columns = [f"mask_{c}" for c in mask_df.columns]
        
        # 💡 왜 마스킹?
        # - 보간 후에는 원래 결측이었는지 알 수 없음
        # - 모델이 "측정 안 됨" 패턴을 학습할 수 있도록
        
        # 4-4. LOCF (Last Observation Carried Forward) 보간
        # 같은 환자 내에서 이전 값으로 채우기
        # 예: time_bucket 6에서 심박수 80 → 7에서 결측 → 7도 80으로 채움
        pivot_df[feature_cols] = pivot_df.groupby('stay_id')[feature_cols].ffill()
        
        # 4-5. 여전히 NaN인 경우 (처음부터 결측) → 0으로 대체
        # 이후 정규화되므로 0은 "평균"을 의미하게 됨
        pivot_df[feature_cols] = pivot_df[feature_cols].fillna(0)

        # -----------------------------------------------------------
        # Step 5: Normalization (Z-score 정규화)
        # -----------------------------------------------------------
        # 각 변수의 스케일을 동일하게 맞춤
        # 예: 심박수(0~200) vs 체온(36~40) → 둘 다 평균 0, 표준편차 1로
        
        self.logger.info("⚖️ Normalizing feature scales...")
        
        for col in feature_cols:
            col_mean = pivot_df[col].mean()
            col_std = pivot_df[col].std() + 1e-8  # 표준편차 0 방지
            
            # Z-score 공식: (x - 평균) / 표준편차
            pivot_df[col] = (pivot_df[col] - col_mean) / col_std
        
        # 💡 정규화 효과:
        # - 모델 학습 속도 향상
        # - 모든 변수에 공정한 가중치 부여
        # - Gradient Descent 안정화

        # -----------------------------------------------------------
        # Step 6: Gender Encoding (범주형 → 숫자)
        # -----------------------------------------------------------
        # 성별을 숫자로 변환: M(남성)=0, F(여성)=1
        if 'gender' in pivot_df.columns:
            pivot_df['gender'] = pivot_df['gender'].map({'M': 0, 'F': 1})

        # -----------------------------------------------------------
        # Step 7: 최종 데이터 결합 및 저장
        # -----------------------------------------------------------
        # 원본 데이터 + 마스킹 데이터 병합
        # 최종 형태:
        # stay_id | time_bucket | age | gender | label | heart_rate | ... | mask_heart_rate | ...
        final_df = pd.concat([pivot_df, mask_df], axis=1)
        
        # Parquet 형식으로 저장 (압축률 높고 빠름)
        output_file = f"{output_dir}/{disease_type}_preprocess.parquet"
        final_df.to_parquet(output_file, index=False)
        
        self.logger.info(f"✅ 저장 완료: {output_file}")
        self.logger.info(f"   데이터 형태: {final_df.shape}")
        self.logger.info(f"   컬럼 수: {len(final_df.columns)}")
        
        return final_df


# ============================================================
# [실행 진입점]
# ============================================================

if __name__ == "__main__":
    # -----------------------------------------------------------
    # 경로 설정
    # -----------------------------------------------------------
    INPUT_DIR = "/home/kido/miniproject/team3/00_extract"   # 추출된 원본 데이터
    OUTPUT_DIR = "/home/kido/miniproject/team3/01_preprocess" # 전처리 결과 저장
    
    # 출력 폴더 자동 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -----------------------------------------------------------
    # 전처리 객체 생성
    # -----------------------------------------------------------
    cleaner = SNU_RealMIP_UnifiedCleaner(VALID_RANGES)

    # -----------------------------------------------------------
    # 질환별 전처리 실행
    # -----------------------------------------------------------
    for disease in ['ami', 'stroke']:
        result = cleaner.process_unified(disease, INPUT_DIR, OUTPUT_DIR)
        
        if result is not None:
            logging.info(f"✅ {disease.upper()} 전처리 완료! 최종 형태: {result.shape}")
            
            # 💡 예상 결과:
            # - AMI: (약 20,000 rows, 35 columns)
            #   → 11 피처 + 11 마스크 + age, gender, label, stay_id, time_bucket
            # - STROKE: (약 12,000 rows, 35 columns)
