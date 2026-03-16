import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, f1_score
from sklearn.calibration import calibration_curve  # 핵심 추가
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/home/kido/miniproject/team3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# [1. 모델 정의 구역]
# ==========================================

# (Ours) MultiModal Model
class MultiModalMIMIC_Focal(nn.Module):
    def __init__(self, time_dim=22, static_dim=20, hidden_dim=64): # static_dim 기본값을 20으로
        super().__init__()
        self.lstm = nn.LSTM(time_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # 체크포인트 사양: 입력 20 -> 출력 16
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 16), # 32에서 16으로 수정
            nn.ReLU(), 
            nn.Dropout(0.2)
        )
        
        # classifier 입력: LSTM(64) + Static(16) = 80 (체크포인트의 1, 80과 일치)
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
    print(f"\n🔥 [{disease_type.upper()}] 8종 모델 벤치마킹 및 Calibration 분석 시작")
    X_time, X_static, y, sids = prepare_clinical_data_original(disease_type)
    
    tr_va_sids, te_sids = train_test_split(np.unique(sids), test_size=0.2, random_state=42)
    train_mask, test_mask = np.isin(sids, tr_va_sids), np.isin(sids, te_sids)
    
    y_test = y[test_mask]
    all_model_probs = {} # Calibration을 위해 확률값 저장

    # --- 1. Ours (MultiModal) ---
    model_ours = MultiModalMIMIC_Focal(static_dim=20, hidden_dim=64).to(device)
    model_path = f"{BASE_DIR}/{disease_type}_lightweight_top20.pth"
    if os.path.exists(model_path):
        model_ours.load_state_dict(torch.load(model_path, map_location=device))
        model_ours.eval()
        with torch.no_grad():
            p = model_ours(torch.FloatTensor(X_time[test_mask]).to(device), 
                           torch.FloatTensor(X_static[test_mask][:, :20]).to(device)).cpu().numpy()
        all_model_probs['Ours (MultiModal)'] = p

    # --- 2. ML Models ---
    X_ml_train, X_ml_test = X_time[train_mask][:, -1, :], X_time[test_mask][:, -1, :]
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
        all_model_probs[name] = model.predict_proba(X_ml_test)[:, 1]

# --- 3. DL Models (Sequential) ---
    dl_models = {
        'Simple LSTM': SimpleLSTM(22),
        'Transformer': TransformerModel(22)
    }

    for name, model in dl_models.items():
        model.to(device)
        # [수정] 이 위치에서 optimizer가 반드시 정의되어야 합니다.
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_ds = TensorDataset(
            torch.FloatTensor(X_time[train_mask]), 
            torch.FloatTensor(y[train_mask])
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        
        model.train()
        for epoch in range(5):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                
                # 이제 여기서 NameError가 발생하지 않습니다.
                optimizer.zero_grad() 
                out = model(bx)
                loss = nn.BCELoss()(out, by)
                loss.backward()
                optimizer.step()
                        
        # 평가 단계에서도 메모리 절약을 위해 평가용 로더 사용 권장
        model.eval()
        p_list = []
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_time[test_mask])), batch_size=256)
        with torch.no_grad():
            for bx, in test_loader:
                p_list.extend(model(bx.to(device)).cpu().numpy())
        all_model_probs[name] = np.array(p_list)

    # ==========================================
    # [4. Calibration Curve 시각화 및 저장]
    # ==========================================
    plt.figure(figsize=(10, 10))
    # 완벽한 모델의 기준선 (y=x)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.6)

    results = []
    for name, probs in all_model_probs.items():
        # 각 모델별 Calibration Curve 계산 (True Probability vs Predicted Probability)
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name)
        
        # 기존 성능 지표 계산
        results.append(calculate_metrics(y_test, probs, name))

    plt.xlabel("Predicted Probability (Mean)", fontsize=12)
    plt.ylabel("True Probability (Fraction of Positives)", fontsize=12)
    plt.title(f"Calibration Curve: {disease_type.upper()}", fontsize=15, fontweight='bold')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # 그래프 저장
    calib_png_path = f"{BASE_DIR}/{disease_type}_calibration_all_models.png"
    plt.savefig(calib_png_path, dpi=300)
    plt.close()
    print(f"✅ Calibration Curve 저장 완료: {calib_png_path}")

    # 결과 표 출력 및 CSV 저장 (기존 로직 유지)
    df_res = pd.DataFrame(results).sort_values('AUROC', ascending=False)
    df_res.to_csv(f"{BASE_DIR}/{disease_type}_benchmark_results.csv", index=False)
    print(f"\n📊 [{disease_type.upper()}] 최종 결과\n", df_res.to_string(index=False))

if __name__ == "__main__":
    for disease in ['ami', 'stroke']:
        run_comprehensive_benchmark(disease)
