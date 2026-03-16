import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, f1_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/home/kido/miniproject/team3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# [1. 모델 정의 구역]
# ==========================================

# (Ours) MultiModal Model - 저장된 파라미터(20차원)에 최적화
class MultiModalMIMIC_Focal(nn.Module):
    def __init__(self, time_dim=22, static_dim=20, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(time_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.static_fc = nn.Sequential(nn.Linear(static_dim, 16), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim + 16, 1))
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
    
    # Static 48차원 규격 생성 (하위 호환 유지)
    X_static = np.zeros((X.shape[0], 48), dtype=np.float32)
    X_static[:, :22] = X[:, -1, :] 
    
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
    print(f"\n🔥 [{disease_type.upper()}] 8종 모델 성능 벤치마킹 및 신뢰도 분석 시작")
    X_time, X_static, y, sids = prepare_clinical_data_original(disease_type)
    
    tr_va_sids, te_sids = train_test_split(np.unique(sids), test_size=0.2, random_state=42)
    train_mask, test_mask = np.isin(sids, tr_va_sids), np.isin(sids, te_sids)
    
    y_test = y[test_mask]
    all_probs = {}
    results = []

    # --- 1. Ours (MultiModal) ---
    model_ours = MultiModalMIMIC_Focal(static_dim=20, hidden_dim=64).to(device)
    model_path = f"{BASE_DIR}/{disease_type}_lightweight_top20.pth"
    if os.path.exists(model_path):
        model_ours.load_state_dict(torch.load(model_path, map_location=device))
        model_ours.eval()
        with torch.no_grad():
            # 입력 시 상위 20개 피처만 슬라이싱하여 RuntimeError 방지
            p = model_ours(torch.FloatTensor(X_time[test_mask]).to(device), 
                           torch.FloatTensor(X_static[test_mask][:, :20]).to(device)).cpu().numpy()
        all_probs['Ours (MultiModal)'] = p
        results.append(calculate_metrics(y_test, p, 'Ours (MultiModal)'))

    # --- 2. ML Models ---
    X_ml_train = X_time[train_mask][:, -1, :]
    X_ml_test = X_time[test_mask][:, -1, :]
    y_ml_train = y[train_mask]

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
        all_probs[name] = p
        results.append(calculate_metrics(y_test, p, name))

    # --- 3. DL Models ---
    dl_models = {'Simple LSTM': SimpleLSTM(22), 'Transformer': TransformerModel(22)}

    for name, model in dl_models.items():
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001) # NameError 방지
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_time[train_mask]), torch.FloatTensor(y[train_mask])), batch_size=256, shuffle=True)
        
        model.train()
        for _ in range(2): # 검증용 2 Epoch
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                nn.BCELoss()(model(bx), by).backward()
                optimizer.step()
        
        model.eval()
        test_p = []
        with torch.no_grad():
            for bx, in DataLoader(TensorDataset(torch.FloatTensor(X_time[test_mask])), batch_size=256):
                test_p.extend(model(bx.to(device)).cpu().numpy())
        all_probs[name] = np.array(test_p)
        results.append(calculate_metrics(y_test, all_probs[name], name))

    # ==========================================
    # [4. 결과 저장 및 시각화]
    # ==========================================
    
    # 1) CSV 저장 (전체 벤치마킹 수치)
    df_res = pd.DataFrame(results).sort_values('AUROC', ascending=False)
    csv_path = f"{BASE_DIR}/{disease_type}_benchmark_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n📊 [{disease_type.upper()}] 벤치마킹 결과\n", df_res.to_string(index=False))
    print(f"✅ CSV 결과 저장 완료: {csv_path}")

    # 2) PNG 저장 (Calibration Curve: Predicted vs True Probability)
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5) # 이상적인 선

    for name, probs in all_probs.items():
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)

    plt.title(f'Calibration Curve (Reliability Diagram) - {disease_type.upper()}', fontsize=15, fontweight='bold')
    plt.xlabel('Predicted Probability (Mean)', fontsize=12)
    plt.ylabel('True Probability (Fraction of Positives)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    png_path = f"{BASE_DIR}/{disease_type}_calibration_curve.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✅ Calibration Curve 시각화 저장 완료: {png_path}")

if __name__ == "__main__":
    for disease in ['ami', 'stroke']:
        run_comprehensive_benchmark(disease)