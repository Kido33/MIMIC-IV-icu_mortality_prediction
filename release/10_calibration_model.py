import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/home/kido/miniproject/team3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# [1. 모델 정의 구역]
# ==========================================

# (Ours) MultiModal Model
class MultiModalMIMIC_Focal(nn.Module):
    def __init__(self, time_dim=22, static_dim=48, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(time_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.static_fc = nn.Sequential(nn.Linear(static_dim, 32), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim + 32, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_time, x_static):
        _, (h_n, _) = self.lstm(x_time)
        combined = torch.cat([h_n[-1], self.static_fc(x_static)], dim=1)
        return self.sigmoid(self.classifier(combined)).squeeze()

# (DL Baseline) Simple LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.sigmoid(self.fc(h_n[-1])).squeeze()

# (DL Baseline) Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim=22, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=128, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.embedding(x); x = self.transformer(x)
        return self.sigmoid(self.fc(x.mean(dim=1))).squeeze()

# ==========================================
# [2. 데이터 및 유틸리티]
# ==========================================

def prepare_clinical_data_original(disease_type):
    X = np.load(f"{BASE_DIR}/{disease_type}_X_rolling.npy") 
    y = np.load(f"{BASE_DIR}/{disease_type}_y_rolling.npy")
    sids = np.load(f"{BASE_DIR}/{disease_type}_sids_rolling.npy")
    
    # Static 48차원 규격 강제 매칭
    X_static = np.zeros((X.shape[0], 48), dtype=np.float32)
    X_static[:, :22] = X[:, -1, :] # 임시 배치
    
    return X.astype(np.float32), X_static, y, sids

def calculate_metrics(y_true, y_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Model': model_name,
        'AUROC': round(roc_auc_score(y_true, y_prob), 4),
        'AUPRC': round(average_precision_score(y_true, y_prob), 4),
        'F1-Score': round(f1_score(y_true, y_pred), 4),
        'ConfusionMatrix(TP/FP/FN/TN)': f"{tp}/{fp}/{fn}/{tn}"
    }

# ==========================================
# [3. 메인 벤치마킹 실행]
# ==========================================

def run_comprehensive_benchmark(disease_type):
    print(f"\n🔥 [{disease_type.upper()}] 8종 모델 벤치마킹 시작")
    X_time, X_static, y, sids = prepare_clinical_data_original(disease_type)
    
    tr_va_sids, te_sids = train_test_split(np.unique(sids), test_size=0.2, random_state=42)
    train_mask, test_mask = np.isin(sids, tr_va_sids), np.isin(sids, te_sids)
    
    results = []

    # --- 1. Ours (MultiModal) ---
    model_ours = MultiModalMIMIC_Focal(static_dim=48).to(device)
    model_path = f"{BASE_DIR}/{disease_type}_light_top20.pth"
    if os.path.exists(model_path):
        model_ours.load_state_dict(torch.load(model_path, map_location=device))
        model_ours.eval()
        with torch.no_grad():
            p = model_ours(torch.FloatTensor(X_time[test_mask]).to(device), 
                           torch.FloatTensor(X_static[test_mask]).to(device)).cpu().numpy()
        results.append(calculate_metrics(y[test_mask], p, 'Ours (MultiModal)'))

    # --- 2. ML Models (Last Time-step) ---
    X_ml_train = X_time[train_mask][:, -1, :]
    X_ml_test = X_time[test_mask][:, -1, :]
    y_ml_train, y_ml_test = y[train_mask], y[test_mask]

    ml_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, verbose=-1),
        'CatBoost': CatBoostClassifier(n_estimators=100, verbose=0)
    }

    for name, model in tqdm(ml_models.items(), desc=" ⚡ ML Benchmarking"):
        model.fit(X_ml_train, y_ml_train)
        p = model.predict_proba(X_ml_test)[:, 1]
        results.append(calculate_metrics(y_ml_test, p, name))

# --- 3. DL Models (Sequential) ---
    dl_models = {
        'Simple LSTM': SimpleLSTM(22),
        'Transformer': TransformerModel(22)
    }

    for name, model in dl_models.items():
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # [핵심] DataLoader를 사용하여 17GB 데이터를 쪼개서 학습
        train_ds = TensorDataset(
            torch.FloatTensor(X_time[train_mask]), 
            torch.FloatTensor(y[train_mask])
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True) # 배치 사이즈 조절
        
        model.train()
        for _ in range(5): # 5 Epoch
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = nn.BCELoss()(out, by)
                loss.backward()
                optimizer.step()
        
        # 평가 단계에서도 메모리 절약을 위해 평가용 로더 사용 권장
        model.eval()
        test_probs = []
        test_ds = TensorDataset(torch.FloatTensor(X_time[test_mask]))
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        
        with torch.no_grad():
            for bx, in test_loader:
                bx = bx.to(device)
                p = model(bx).cpu().numpy()
                test_probs.extend(p)
        
        results.append(calculate_metrics(y[test_mask], np.array(test_probs), name))
        
        # 메모리 해제
        del train_ds, train_loader, test_ds, test_loader
        torch.cuda.empty_cache()
        
# --- 결과 요약 및 저장 ---
    df_res = pd.DataFrame(results).sort_values('AUROC', ascending=False)
    print(f"\n📊 [{disease_type.upper()}] 8종 비교 결과\n", df_res.to_string(index=False))
    
    # 1. CSV 저장 (데이터 보존용)
    csv_path = f"{BASE_DIR}/{disease_type}_benchmark_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"✅ CSV 결과 저장 완료: {csv_path}")

    # 2. PNG 시각화 및 저장 (꺾은선 그래프 버전)
    plt.figure(figsize=(14, 7))
    
    # AUROC 꺾은선
    plt.plot(df_res['Model'], df_res['AUROC'], marker='o', linestyle='-', 
             linewidth=2, markersize=8, color='#1f77b4', label='AUROC')
    
    # AUPRC 꺾은선
    plt.plot(df_res['Model'], df_res['AUPRC'], marker='s', linestyle='--', 
             linewidth=2, markersize=8, color='#ff7f0e', label='AUPRC')
    
    # 수치 텍스트 표시 (가독성 향상)
    for i, row in df_res.iterrows():
        plt.text(i, row['AUROC'] + 0.02, f"{row['AUROC']:.3f}", ha='center', va='bottom', fontsize=10, color='#1f77b4')
        plt.text(i, row['AUPRC'] - 0.04, f"{row['AUPRC']:.3f}", ha='center', va='top', fontsize=10, color='#ff7f0e')

    plt.title(f'Performance Trend by Model - {disease_type.upper()}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Metric Score', fontsize=12)
    plt.xlabel('Models (Ranked by AUROC)', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0, 1.1) # 여유 공간 확보
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    png_path = f"{BASE_DIR}/{disease_type}_performance_trend.png"
    plt.savefig(png_path, dpi=300) # 고해상도 저장
    plt.close()
    print(f"✅ 꺾은선 그래프 저장 완료: {png_path}")

if __name__ == "__main__":
    for disease in ['ami', 'stroke']:
        run_comprehensive_benchmark(disease)
