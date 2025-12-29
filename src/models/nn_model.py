import numpy as np
import pandas as pd
import torch
import io
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path
import time
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = Path("models/nn_model.pth")
SCALER_PATH = Path("models/nn_scaler.joblib")

class CreditScoringNN(nn.Module):
    """Нейронная сеть для кредитного скоринга"""
    def __init__(self, input_size: int, dropout_rate: float = 0.3):
        super(CreditScoringNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def prepare_data_for_nn(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
    """Подготовка данных для нейронной сети с обработкой NaN"""
    print("\nPreprocessing data...")
    
    # Копируем и заполняем NaN
    X_clean = X.copy()
    
    # Заполняем NaN медианами по колонкам
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            median_val = X_clean[col].median()
            X_clean[col] = X_clean[col].fillna(median_val)
            print(f"  Filled NaN in {col} with median: {median_val:.4f}")
    
    # Используем RobustScaler вместо StandardScaler (устойчивее к выбросам)
    scaler = RobustScaler(quantile_range=(25, 75))
    X_scaled = scaler.fit_transform(X_clean)
    
    # Проверяем на NaN после масштабирования
    if np.isnan(X_scaled).any():
        print(f"Warning: NaN after scaling, filling with 0")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    print(f"  Data shape: {X_scaled.shape}")
    print(f"  Scaled data - Min: {X_scaled.min():.4f}, Max: {X_scaled.max():.4f}")
    print(f"  Scaled data - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    
    return X_scaled, y.values, scaler

def safe_clip_predictions(predictions: torch.Tensor) -> torch.Tensor:
    """Безопасное ограничение предсказаний"""
    # Ограничиваем очень маленькими значениями, но не 0 и 1
    return torch.clamp(predictions, 1e-7, 1 - 1e-7)

def train_nn_model(X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   input_size: int, 
                   epochs: int = 30,
                   batch_size: int = 128,
                   learning_rate: float = 0.001) -> Tuple[CreditScoringNN, dict]:
    """Обучение нейронной сети с улучшенной стабильностью"""
    
    print(f"\nTraining setup:")
    print(f"  Input size: {input_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Создание даталоадеров
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).reshape(-1, 1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Инициализация модели с инициализацией весов
    model = CreditScoringNN(input_size=input_size)
    
    # Xavier инициализация весов для лучшей стабильности
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Обучение
    train_losses = []
    val_losses = []
    val_auc_scores = []
    
    print("\nTraining progress:")
    print("-" * 70)
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Тренировка
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # Безопасное ограничение предсказаний
            predictions = safe_clip_predictions(predictions)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                predictions = safe_clip_predictions(predictions)
                
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                val_predictions.extend(predictions.numpy())
                val_targets.extend(batch_y.numpy())
        
        # Расчет метрик
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Обработка предсказаний перед расчетом AUC
        val_predictions_np = np.array(val_predictions).flatten()
        val_targets_np = np.array(val_targets).flatten()
        
        # Проверка корректности
        if np.any(np.isnan(val_predictions_np)):
            print(f"Epoch {epoch+1}: Warning - NaN in predictions, skipping AUC calculation")
            val_auc = 0.5  # Случайный классификатор
        else:
            try:
                val_auc = roc_auc_score(val_targets_np, val_predictions_np)
            except:
                val_auc = 0.5
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_auc_scores.append(val_auc)
        
        # Сохранение лучшей модели
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Выводим прогресс каждую эпоху
        print(f'Epoch [{epoch+1:3d}/{epochs}] | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val AUC: {val_auc:.4f} | '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Ранняя остановка
        if epoch > 10:
            # Если AUC падает 3 эпохи подряд
            if all(val_auc_scores[-i] < val_auc_scores[-i-1] for i in range(1, 4)):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Загружаем лучшие веса
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_auc_scores': val_auc_scores,
        'best_val_auc': best_val_auc,
        'final_val_auc': val_auc_scores[-1] if val_auc_scores else 0
    }
    
    return model, metrics

def main():
    """Основная функция для обучения NN модели"""
    from src.models.train import load_data
    
    print("="*70)
    print("TRAINING NEURAL NETWORK FOR CREDIT SCORING")
    print("="*70)
    
    # Загрузка данных
    print("\nLoading data...")
    X, y = load_data()
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: 0={sum(y==0)} ({sum(y==0)/len(y):.2%}), "
          f"1={sum(y==1)} ({sum(y==1)/len(y):.2%})")
    
    # Проверяем NaN в исходных данных
    print(f"\nChecking data quality...")
    nan_counts = X.isna().sum()
    if nan_counts.any():
        print(f"Found NaN in columns:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"{col}: {nan_counts[col]} NaN values")
    else:
        print(f"No NaN values found")
    
    # Разделение данных
    print(f"\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Data split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples") 
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # Подготовка данных для NN
    X_train_scaled, y_train_array, scaler = prepare_data_for_nn(X_train, y_train)
    X_val_scaled, y_val_array, _ = prepare_data_for_nn(X_val, y_val)
    X_test_scaled, y_test_array, _ = prepare_data_for_nn(X_test, y_test)
    
    # Обучение модели
    input_size = X_train_scaled.shape[1]
    print(f"\n" + "="*70)
    
    model, metrics = train_nn_model(
        X_train_scaled, y_train_array,
        X_val_scaled, y_val_array,
        input_size=input_size,
        epochs=40,
        batch_size=256,
        learning_rate=0.0003
    )
    
    # Оценка на тестовых данных
    print(f"\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_pred_proba = model(X_test_tensor).numpy().flatten()
        y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
    
    test_auc = roc_auc_score(y_test_array, y_pred_proba)
    
    # Расчет accuracy
    y_pred = (y_pred_proba >= 0.5).astype(int)
    test_accuracy = np.mean(y_pred == y_test_array)
    
    print(f"\nTest Results:")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Best Val AUC: {metrics['best_val_auc']:.4f}")
    
    # Сохранение модели и скейлера
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'test_auc': test_auc,
        'test_accuracy': test_accuracy,
        'best_val_auc': metrics['best_val_auc'],
        'architecture': '32-16-8-1',
        'scaler_mean': scaler.center_ if hasattr(scaler, 'center_') else None,
        'scaler_scale': scaler.scale_ if hasattr(scaler, 'scale_') else None
    }, MODEL_PATH)
    
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"\nSaved files:")
    print(f"Model:      {MODEL_PATH}")
    print(f"Scaler:     {SCALER_PATH}")
    
    # Сохранение метрик
    metrics_path = Path("models/nn_model_metrics.json")
    import json
    with open(metrics_path, 'w') as f:
        json.dump({
            'test_auc': float(test_auc),
            'test_accuracy': float(test_accuracy),
            'best_val_auc': float(metrics['best_val_auc']),
            'input_size': int(input_size),
            'architecture': '32-16-8-1',
            'training_samples': int(len(X_train)),
            'validation_samples': int(len(X_val)),
            'test_samples': int(len(X_test))
        }, f, indent=2)
    
    print(f"Metrics:    {metrics_path}")
    
    # Визуализация кривой обучения
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_losses'], label='Train Loss')
        plt.plot(metrics['val_losses'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(metrics['val_auc_scores'], label='Val AUC', color='green')
        plt.axhline(y=test_auc, color='red', linestyle='--', label='Test AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Validation AUC over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/nn_training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training plot: models/nn_training_history.png")
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    print(f"\n" + "="*70)
    print("NEURAL NETWORK TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return model, scaler, test_auc

if __name__ == "__main__":
    main()