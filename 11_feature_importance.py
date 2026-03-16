import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import os
import matplotlib.pyplot as plt

# [1. 전역 설정]
BASE_DIR = "/home/kido/miniproject/team3"
device = torch.device("cpu")

# [2. 모델 및 Wrapper 정의]
class MultiModalMIMIC(nn.Module):
    def __init__(self, time_dim=23, static_dim=4, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(time_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.static_fc = nn.Sequential(nn.Linear(static_dim, 16), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim + 16, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x_time, x_static):
        _, (h_n, _) = self.lstm(x_time)
        time_feat = h_n[-1]
        static_feat = self.static_fc(x_static)
        combined = torch.cat([time_feat, static_feat], dim=1)
        return self.classifier(combined).squeeze()

class ModelWrapper(nn.Module):
    def __init__(self, model, hr_indices):
        super().__init__()
        self.model = model
        self.hr_indices = hr_indices
    def forward(self, x_refined):
        batch_size = x_refined.shape[0]
        x_full = torch.zeros((batch_size, 142), dtype=torch.float32)
        non_hr_indices = [i for i in range(142) if i not in self.hr_indices]
        x_full[:, non_hr_indices] = torch.as_tensor(x_refined, dtype=torch.float32)
        x_time = x_full[:, :6*23].reshape(-1, 6, 23)
        x_static = x_full[:, 6*23:]
        return self.model(x_time, x_static).view(-1, 1)

# [3. 메인 분석 함수]
def run_final_no_hr_analysis(disease_type):
    print(f"\n🧬 [{disease_type.upper()}] 통합 분석(HR 제외) 프로세스 가동...")
    
    # 데이터 로딩
    try:
        X_raw = np.load(f"{BASE_DIR}/{disease_type}_X_rolling.npy")
        sids = np.load(f"{BASE_DIR}/{disease_type}_sids_rolling.npy")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 데이터 전처리 (Rolling Window 정렬 및 Slope 추가)
    X_shifted = X_raw[:-1]
    valid_mask = sids[:-1] == sids[1:]
    X_final = X_shifted[valid_mask]
    
    slopes = (X_final[:, -1, 0] - X_final[:, 0, 0]) / 5.0
    slopes_exp = np.repeat(slopes[:, np.newaxis, np.newaxis], 6, axis=1)
    X_time = np.concatenate([X_final, slopes_exp], axis=2)
    X_static = X_final[:, 0, :4]
    X_combined = np.hstack([X_time.reshape(X_time.shape[0], -1), X_static])

    vitals = ['HR', 'SBP', 'DBP', 'MBP', 'RR', 'Temp', 'SpO2', 'FiO2', 'GCS_E', 'GCS_V', 'GCS_M', 
              'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'HR_Delta']
    feature_names = [f"{v}_t{t}" for t in range(6) for v in vitals] + ['Age', 'Gender', 'Height', 'Weight']

    # HR 제외 인덱스 추출
    hr_indices = [i for i, name in enumerate(feature_names) if 'HR' in name]
    non_hr_indices = [i for i, name in enumerate(feature_names) if 'HR' not in name]
    refined_feature_names = [feature_names[i] for i in non_hr_indices]
    X_refined = X_combined[:, non_hr_indices]

    # 모델 로드
    model = MultiModalMIMIC().to(device)
    model_path = f"{BASE_DIR}/{disease_type}_fixed_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    wrapper = ModelWrapper(model, hr_indices)

    # SHAP 연산 (백그라운드 100, 타겟 200 샘플링)
    bg_idx = np.random.choice(len(X_refined), 100, replace=False)
    tg_idx = np.random.choice(len(X_refined), 200, replace=False)
    explainer = shap.GradientExplainer(wrapper, torch.FloatTensor(X_refined[bg_idx]))
    shap_values = explainer.shap_values(torch.FloatTensor(X_refined[tg_idx]))
    if isinstance(shap_values, list): shap_values = shap_values[0]

    # [중요] 지표 그룹화 로직
    group_map = {}
    for i, name in enumerate(refined_feature_names):
        group_name = name.split('_t')[0] if '_t' in name else name
        if group_name not in group_map: group_map[group_name] = []
        group_map[group_name].append(i)

    group_names = list(group_map.keys())
    grouped_shap = np.stack([shap_values[:, group_map[g]].sum(axis=1) for g in group_names], axis=1)
    grouped_data = np.stack([X_refined[tg_idx][:, group_map[g]].mean(axis=1) for g in group_names], axis=1)

    # --- [데이터 정렬 및 상위 10개 추출] ---
    total_imp = np.abs(grouped_shap).mean(axis=0).flatten()
    sorted_inds = np.argsort(total_imp)[::-1].astype(int)
    
    top_10_indices = sorted_inds[:10]
    top_10_names = [group_names[i] for i in top_10_indices]
    top_10_shap_values = grouped_shap[:, top_10_indices]
    top_10_grouped_data = grouped_data[:, top_10_indices]

    # --- [시각화 1: Beeswarm Plot] ---
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
    top_10_shap_values,   # 상위 10개 데이터만 포함됨
    top_10_grouped_data, 
    feature_names=top_10_names, 
    max_display=10,       # 10개 출력 강제
    show=False
)
    plt.title(f"Top 10 Feature Impact (Beeswarm): {disease_type.upper()}", fontsize=15)
    plt.savefig(f"{BASE_DIR}/{disease_type}_top10_beeswarm.png", bbox_inches='tight')
    plt.close()

    # --- [시각화 2: Bar Plot] ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(top_10_shap_values, top_10_grouped_data, feature_names=top_10_names, plot_type='bar', max_display=10, show=False)
    plt.title(f"Top 10 Feature Importance (Bar Chart): {disease_type.upper()}", fontsize=15)
    plt.savefig(f"{BASE_DIR}/{disease_type}_top10_bar.png", bbox_inches='tight')
    plt.close()

    # --- [통계 데이터 저장] ---
    res_df = pd.DataFrame({
        'Feature': [group_names[i] for i in sorted_inds],
        'Importance': total_imp[sorted_inds]
    })
    res_df.to_csv(f"{BASE_DIR}/{disease_type}_no_hr_importance.csv", index=False)
    
    print(f"📊 {disease_type.upper()} 분석 완료. 그래프 2종 및 CSV 저장 완료.")

if __name__ == "__main__":
    for d in ['ami', 'stroke']:
        run_final_no_hr_analysis(d)
    print("\n🚀 모든 작업이 성공적으로 완료되었습니다.")