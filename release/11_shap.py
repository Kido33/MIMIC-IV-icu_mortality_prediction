import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# [1. 경로 설정] - 현재 디렉토리와 BASE_DIR을 모두 체크하도록 설정
BASE_DIR = "/home/kido/miniproject/team3" 
filenames = {
    'ami': {
        'values': 'ami_shap_values.npy',
        'data': 'ami_shap_data.npy',
        'names': 'ami_shap_feature_names.npy'
    },
    'stroke': {
        'values': 'stroke_shap_values.npy',
        'data': 'stroke_shap_data.npy',
        'names': 'stroke_shap_feature_names.npy'
    }
}

def draw_real_beeswarm(disease_type):
    print(f"🧬 [{disease_type.upper()}] Beeswarm 분석 및 시각화 시작...")
    
    f_info = filenames[disease_type]
    
    # 파일 존재 여부 확인 후 로드
    try:
        # 파일이 현재 폴더에 없다면 BASE_DIR에서 찾음
        v_path = f_info['values'] if os.path.exists(f_info['values']) else os.path.join(BASE_DIR, f_info['values'])
        d_path = f_info['data'] if os.path.exists(f_info['data']) else os.path.join(BASE_DIR, f_info['data'])
        n_path = f_info['names'] if os.path.exists(f_info['names']) else os.path.join(BASE_DIR, f_info['names'])
        
        shap_values = np.load(v_path)
        data = np.load(d_path)
        feature_names = np.load(n_path)
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인하세요: {e}")
        return

    # [차원 정렬] (500, 142, 1) -> (500, 142)
    if len(shap_values.shape) == 3:
        shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

    # [지표 그룹화] (t0~t5 통합 시각화)
    group_map = {}
    for i, name in enumerate(feature_names):
        base_name = name.split('_t')[0] if '_t' in name else name
        if 'HR' in base_name: continue # HR 제외
        if base_name not in group_map: group_map[base_name] = []
        group_map[base_name].append(i)

    group_names = list(group_map.keys())
    
    # 그룹별 SHAP 합산 및 원본 데이터 평균 계산
    grouped_shap = np.stack([shap_values[:, group_map[g]].sum(axis=1) for g in group_names], axis=1)
    grouped_data = np.stack([data[:, group_map[g]].mean(axis=1) for g in group_names], axis=1)

    # [상위 10개 정렬]
    mean_abs_shap = np.abs(grouped_shap).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1] # 내림차순
    
    top_10_idx = sorted_indices[:10]
    top_10_shap = grouped_shap[:, top_10_idx]
    top_10_data = grouped_data[:, top_10_idx]
    top_10_names = [group_names[i] for i in top_10_idx]

    # [시각화]
    plt.figure(figsize=(12, 10))
    # 시각적 가독성을 위해 정렬된 순서대로 표시
    shap.summary_plot(
        top_10_shap, 
        top_10_data, 
        feature_names=top_10_names,
        max_display=10,
        plot_type="dot", # Beeswarm
        show=False
    )
    
    plt.title(f"SHAP Beeswarm Plot: {disease_type.upper()} (Top 10 Features)", fontsize=16)
    plt.tight_layout()
    
    save_name = f"{disease_type}.png"
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"✅ {disease_type.upper()} Beeswarm 그래프 저장 완료: {save_name}")

if __name__ == "__main__":
    for d in ['ami', 'stroke']:
        draw_real_beeswarm(d)
    print("\n🚀 모든 분석이 완료되었습니다.")
