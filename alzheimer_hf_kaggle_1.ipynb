"""
╔══════════════════════════════════════════════════════════════╗
║   Alzheimer MRI 4-Class Classification — Kaggle Version      ║
║   Metric  : Macro F1                                         ║
║   Problem : 극심한 클래스 불균형 (220:1)                      ║
║   Strategy:                                                  ║
║     1) Focal Loss       — 어려운 샘플 집중 학습               ║
║     2) WeightedSampler  — ModerateDemented 오버샘플링         ║
║     3) timm EfficientNet-B4 (pretrained)                     ║
║     4) Stratified K-Fold + OOF 앙상블                        ║
║     5) Threshold 최적화 — Macro F1 Post-processing            ║
║     6) TTA (Test-Time Augmentation)                          ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────
# [STEP 0] 패키지 설치 (최초 1회)
# ─────────────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "-q"]
    )

REQUIRED = {
    "timm"        : "timm",
    "sklearn"     : "scikit-learn",
    "tqdm"        : "tqdm",
    "PIL"         : "Pillow",
    "torch"       : "torch",
    "torchvision" : "torchvision",
}
for import_name, pkg_name in REQUIRED.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"[설치] {pkg_name} ...")
        install(pkg_name)

# ─────────────────────────────────────────────────────────────
# [STEP 1] 라이브러리 임포트
# ─────────────────────────────────────────────────────────────
import os, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import torchvision.transforms as T
import timm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# [STEP 2] 전역 설정
# ─────────────────────────────────────────────────────────────
CFG = {
    # ── 경로 ──────────────────────────────────────────────────
    "train_csv"            : "/kaggle/input/competitions/alzheimer-prediction/data_V2/train.csv",
    "sample_submission_csv": "/kaggle/input/competitions/alzheimer-prediction/data_V2/sample_submission.csv",
    "train_img_dir"        : "/kaggle/input/competitions/alzheimer-prediction/data_V2/train/",
    "test_img_dir"         : "/kaggle/input/competitions/alzheimer-prediction/data_V2/test/",
    "output_dir"   : "/kaggle/working/output",

    # ── 모델 ──────────────────────────────────────────────────
    "model_name"   : "efficientnet_b4",   # timm 모델명
    "img_size"     : 224,                 # 256으로 올리면 성능↑ (메모리↑)
    "num_classes"  : 4,
    "pretrained"   : True,
    "drop_rate"    : 0.4,                 # dropout

    # ── 학습 ──────────────────────────────────────────────────
    "batch_size"   : 32,
    "num_epochs"   : 15,
    "lr"           : 2e-4,
    "min_lr"       : 1e-6,
    "weight_decay" : 1e-4,
    "num_folds"    : 3,
    "patience"     : 5,                   # Early Stopping
    "seed"         : 42,

    # ── Focal Loss ────────────────────────────────────────────
    "focal_alpha"  : None,                # None → 클래스 가중치 자동 계산
    "focal_gamma"  : 2.0,                 # γ=2 표준값 (높을수록 어려운 샘플 집중)

    # ── Sampler ───────────────────────────────────────────────
    "use_sampler"  : True,                # WeightedRandomSampler 사용 여부
    "sampler_beta" : 0.9999,              # Class-Balanced Sampling β

    # ── TTA ───────────────────────────────────────────────────
    "tta_times"    : 1,                   # Test-Time Augmentation 반복 횟수

    # ── 기타 ──────────────────────────────────────────────────
    "num_workers"  : 2,  # Kaggle CPU 코어 제한
    "device"       : "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp"      : torch.cuda.is_available(),
    "label_smoothing": 0.05,
}

# 클래스 정의 (고정)
CLASSES    = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS2IDX  = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS  = {i: c for c, i in CLASS2IDX.items()}

# ─────────────────────────────────────────────────────────────
# [STEP 3] 유틸리티
# ─────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def print_class_distribution(df: pd.DataFrame, title: str = ""):
    counts = df["label"].value_counts()
    total  = len(df)
    print(f"\n{'─'*45}")
    if title:
        print(f"  {title}")
    for cls in CLASSES:
        cnt = counts.get(cls, 0)
        bar = "█" * int(cnt / total * 40)
        print(f"  {cls:<22} {cnt:>6,}  ({cnt/total*100:5.1f}%)  {bar}")
    print(f"{'─'*45}")


# ─────────────────────────────────────────────────────────────
# [STEP 4] Class-Balanced Sampling 가중치
# ─────────────────────────────────────────────────────────────
def get_sample_weights(labels: list, beta: float = 0.9999) -> list:
    """
    Class-Balanced Sampling (CVPR 2019)
    weight_i = (1 - β) / (1 - β^n_i)
    → 소수 클래스에 높은 샘플링 확률 부여
    """
    counter = Counter(labels)
    weights_per_class = {}
    for cls, n in counter.items():
        weights_per_class[cls] = (1.0 - beta) / (1.0 - beta ** n)

    # 정규화
    total_w = sum(weights_per_class.values())
    for cls in weights_per_class:
        weights_per_class[cls] /= total_w

    return [weights_per_class[lbl] for lbl in labels]


def get_class_weights_for_loss(labels: list, beta: float = 0.9999) -> torch.Tensor:
    """Focal Loss의 alpha 가중치 계산"""
    counter = Counter(labels)
    weights = []
    for cls in CLASSES:
        n = counter.get(cls, 1)
        w = (1.0 - beta) / (1.0 - beta ** n)
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * len(CLASSES)   # 스케일 정규화
    return weights


# ─────────────────────────────────────────────────────────────
# [STEP 5] Focal Loss
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    γ (gamma) : focusing parameter — 높을수록 어려운 샘플 가중치 ↑
    α (alpha) : 클래스별 가중치 텐서
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)

        # Label Smoothing 적용
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        # Focal weight: (1 - p_t)^γ
        p_t = (probs * smooth_targets).sum(dim=1)
        focal_weight = (1.0 - p_t) ** self.gamma

        # Cross-entropy with smooth targets
        ce_loss = -(smooth_targets * log_probs).sum(dim=1)

        # Alpha 가중치
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = (alpha * smooth_targets).sum(dim=1)
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────
# [STEP 6] Dataset
# ─────────────────────────────────────────────────────────────
class AlzheimerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str,
                 transform=None, is_test: bool = False):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform
        self.is_test   = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.img_dir, row["filename"])

        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # 손상 이미지 대비 fallback
            image = Image.new("RGB", (CFG["img_size"], CFG["img_size"]), 0)

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row["filename"]

        label = CLASS2IDX[row["label"]]
        return image, label


# ─────────────────────────────────────────────────────────────
# [STEP 7] Transforms
# ─────────────────────────────────────────────────────────────
def get_transforms(phase: str, img_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if phase == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=20),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Cutout 효과
        ])
    elif phase == "tta":
        # TTA용: 가벼운 랜덤 변환
        return T.Compose([
            T.Resize((img_size + 16, img_size + 16)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:   # val / test
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


# ─────────────────────────────────────────────────────────────
# [STEP 8] 모델 빌드 (timm)
# ─────────────────────────────────────────────────────────────
class AlzheimerModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int,
                 pretrained: bool, drop_rate: float):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained   = pretrained,
            num_classes  = 0,           # head 제거
            drop_rate    = drop_rate,
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


# ─────────────────────────────────────────────────────────────
# [STEP 9] Mixup Augmentation
# ─────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    """Mixup: 두 샘플을 λ 비율로 섞어서 모델 정규화"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index      = torch.randperm(batch_size, device=x.device)
    mixed_x    = lam * x + (1 - lam) * x[index]
    y_a, y_b   = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────
# [STEP 10] 학습 루프
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    device, use_amp, use_mixup=True):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  [Train]", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup (50% 확률)
        do_mixup = use_mixup and random.random() < 0.5
        if do_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            if do_mixup:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        # Gradient Clipping — 학습 안정화
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        # Mixup 시엔 원본 label(labels_a) 기준으로 로깅
        log_labels = labels_a if do_mixup else labels
        all_labels.extend(log_labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="  [Valid]", leave=False,
                                bar_format="{l_bar}{bar:20}{r_bar}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, np.array(all_probs), np.array(all_labels)


# ─────────────────────────────────────────────────────────────
# [STEP 11] Threshold 최적화 (Macro F1 극대화)
# ─────────────────────────────────────────────────────────────
def optimize_thresholds(probs: np.ndarray, true_labels: np.ndarray,
                        n_search: int = 200) -> np.ndarray:
    """
    클래스별 softmax 임계값을 조정해 Macro F1을 최대화
    OOF 예측 확률을 사용해 최적 threshold 탐색
    """
    best_f1     = 0.0
    best_thresh = np.ones(CFG["num_classes"]) * 0.25

    for _ in range(n_search):
        # 각 클래스에 랜덤 threshold 시도
        thresh = np.random.dirichlet(np.ones(CFG["num_classes"]))

        # threshold 적용: 조정된 확률로 클래스 결정
        adjusted  = probs / (thresh + 1e-9)
        preds     = adjusted.argmax(axis=1)
        f1        = f1_score(true_labels, preds, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh.copy()

    print(f"\n  ✅ Threshold 최적화 완료")
    print(f"     최적 Macro F1: {best_f1:.4f}")
    for i, cls in enumerate(CLASSES):
        print(f"     {cls:<22}: {best_thresh[i]:.4f}")
    return best_thresh


def apply_threshold(probs: np.ndarray, thresh: np.ndarray) -> np.ndarray:
    adjusted = probs / (thresh + 1e-9)
    return adjusted.argmax(axis=1)


# ─────────────────────────────────────────────────────────────
# [STEP 12] TTA 추론
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, df, img_dir, device, use_amp, tta_times: int = 5):
    """
    TTA: val transform 1회 + tta transform (tta_times-1)회 → softmax 평균
    """
    model.eval()
    all_filenames = df["filename"].tolist()
    final_probs   = np.zeros((len(df), CFG["num_classes"]))

    transforms_list = (
        [get_transforms("val", CFG["img_size"])] +
        [get_transforms("tta", CFG["img_size"]) for _ in range(tta_times - 1)]
    )

    for t_idx, transform in enumerate(transforms_list):
        dataset = AlzheimerDataset(df, img_dir, transform, is_test=True)
        loader  = DataLoader(
            dataset, batch_size=CFG["batch_size"], shuffle=False,
            num_workers=CFG["num_workers"], pin_memory=True,
        )
        fold_probs = []
        for images, _ in tqdm(loader,
                               desc=f"  [TTA {t_idx+1}/{len(transforms_list)}]",
                               leave=False, bar_format="{l_bar}{bar:20}{r_bar}"):
            images = images.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            fold_probs.append(probs)

        final_probs += np.vstack(fold_probs) / len(transforms_list)

    return final_probs, all_filenames


# ─────────────────────────────────────────────────────────────
# [STEP 13] 메인 파이프라인
# ─────────────────────────────────────────────────────────────
def main():
    seed_everything(CFG["seed"])
    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = CFG["device"]

    print(f"\n{'═'*55}")
    print(f"  Alzheimer MRI Classification Pipeline")
    print(f"  Device : {device}  |  AMP : {CFG['use_amp']}")
    print(f"  Model  : {CFG['model_name']}  |  img_size : {CFG['img_size']}")
    print(f"{'═'*55}")

    # ── 데이터 로드 ─────────────────────────────────────────
    train_df = pd.read_csv(CFG["train_csv"])

    # test.csv 없음 → sample_submission.csv의 filename 컬럼 사용
    sample_df = pd.read_csv(CFG["sample_submission_csv"])
    test_df   = sample_df[["filename"]].copy()

    print_class_distribution(train_df, "학습 데이터 클래스 분포")
    print(f"\n  Train: {len(train_df):,}  |  Test: {len(test_df):,}  (sample_submission.csv 기준)")

    # 클래스 가중치 (Focal Loss alpha)
    train_labels_raw  = train_df["label"].tolist()
    class_weight_vec  = get_class_weights_for_loss(train_labels_raw, CFG["sampler_beta"])
    print(f"\n  Focal Loss α: {dict(zip(CLASSES, class_weight_vec.numpy().round(4)))}")

    # ── K-Fold 초기화 ────────────────────────────────────────
    skf        = StratifiedKFold(n_splits=CFG["num_folds"], shuffle=True,
                                 random_state=CFG["seed"])
    oof_probs  = np.zeros((len(train_df), CFG["num_classes"]))
    test_probs = np.zeros((len(test_df),  CFG["num_classes"]))
    fold_best_f1s = []

    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["label"]), start=1
    ):
        print(f"\n{'═'*55}")
        print(f"  FOLD {fold} / {CFG['num_folds']}")
        print(f"{'═'*55}")

        tr_df  = train_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # ── Transforms ──────────────────────────────────────
        tr_transform  = get_transforms("train", CFG["img_size"])
        val_transform = get_transforms("val",   CFG["img_size"])

        tr_dataset  = AlzheimerDataset(tr_df,  CFG["train_img_dir"], tr_transform)
        val_dataset = AlzheimerDataset(val_df, CFG["train_img_dir"], val_transform)

        # ── WeightedRandomSampler ────────────────────────────
        if CFG["use_sampler"]:
            sample_weights = get_sample_weights(
                tr_df["label"].tolist(), CFG["sampler_beta"]
            )
            sampler = WeightedRandomSampler(
                weights     = sample_weights,
                num_samples = len(tr_df),
                replacement = True,
            )
            tr_loader = DataLoader(
                tr_dataset, batch_size=CFG["batch_size"],
                sampler=sampler, num_workers=CFG["num_workers"],
                pin_memory=True, drop_last=True,
            )
        else:
            tr_loader = DataLoader(
                tr_dataset, batch_size=CFG["batch_size"],
                shuffle=True, num_workers=CFG["num_workers"],
                pin_memory=True, drop_last=True,
            )

        val_loader = DataLoader(
            val_dataset, batch_size=CFG["batch_size"],
            shuffle=False, num_workers=CFG["num_workers"], pin_memory=True,
        )

        # ── 모델 / 손실 / 옵티마이저 ─────────────────────────
        model = AlzheimerModel(
            CFG["model_name"], CFG["num_classes"],
            CFG["pretrained"], CFG["drop_rate"]
        ).to(device)

        criterion = FocalLoss(
            alpha           = class_weight_vec,
            gamma           = CFG["focal_gamma"],
            label_smoothing = CFG["label_smoothing"],
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr           = CFG["lr"],
            weight_decay = CFG["weight_decay"],
        )

        # Warmup + Cosine Annealing
        warmup_epochs = 3
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=warmup_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max   = CFG["num_epochs"] - warmup_epochs,
                    eta_min = CFG["min_lr"],
                ),
            ],
            milestones=[warmup_epochs],
        )

        scaler    = GradScaler(enabled=CFG["use_amp"])
        best_f1   = 0.0
        no_improv = 0
        best_path = os.path.join(CFG["output_dir"], f"fold{fold}_best.pth")

        # ── Epoch 루프 ───────────────────────────────────────
        for epoch in range(1, CFG["num_epochs"] + 1):
            use_mixup = epoch > warmup_epochs   # warmup 중엔 Mixup 비활성화

            tr_loss, tr_f1 = train_one_epoch(
                model, tr_loader, optimizer, criterion,
                scaler, device, CFG["use_amp"], use_mixup=use_mixup
            )
            val_loss, val_f1, val_probs_ep, val_true_ep = validate(
                model, val_loader, criterion, device, CFG["use_amp"]
            )
            scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]

            flag = ""
            if val_f1 > best_f1:
                best_f1   = val_f1
                no_improv = 0
                torch.save({"model": model.state_dict(),
                            "epoch": epoch,
                            "f1"   : best_f1}, best_path)
                flag = "  ★ BEST"

                # OOF 확률 저장 (best epoch 기준)
                oof_probs[val_idx] = val_probs_ep
            else:
                no_improv += 1

            print(f"  Ep[{epoch:02d}/{CFG['num_epochs']}] "
                  f"lr={curr_lr:.2e}  "
                  f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f}  |  "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}{flag}")

            if no_improv >= CFG["patience"]:
                print(f"  ⏹ Early Stopping (patience={CFG['patience']})")
                break

        fold_best_f1s.append(best_f1)
        print(f"\n  Fold {fold} 최고 Val F1: {best_f1:.4f}")

        # ── 테스트 추론 (TTA 포함) ────────────────────────────
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        fold_test_probs, test_filenames = predict_with_tta(
            model, test_df, CFG["test_img_dir"],
            device, CFG["use_amp"], CFG["tta_times"]
        )
        test_probs += fold_test_probs / CFG["num_folds"]

    # ─────────────────────────────────────────────────────────
    # [STEP 14] OOF 전체 평가
    # ─────────────────────────────────────────────────────────
    oof_true       = train_df["label"].map(CLASS2IDX).values
    oof_preds_raw  = oof_probs.argmax(axis=1)
    oof_f1_raw     = f1_score(oof_true, oof_preds_raw, average="macro", zero_division=0)

    print(f"\n{'═'*55}")
    print(f"  K-Fold 결과 요약")
    print(f"{'═'*55}")
    for i, f1 in enumerate(fold_best_f1s, 1):
        print(f"  Fold {i}: {f1:.4f}")
    print(f"  평균 : {np.mean(fold_best_f1s):.4f} ± {np.std(fold_best_f1s):.4f}")
    print(f"  OOF Macro F1 (기본 threshold): {oof_f1_raw:.4f}")

    # ─────────────────────────────────────────────────────────
    # [STEP 15] Threshold 최적화
    # ─────────────────────────────────────────────────────────
    print(f"\n  Threshold 최적화 탐색 중 (n=200)...")
    best_thresh = optimize_thresholds(oof_probs, oof_true, n_search=200)

    oof_preds_opt = apply_threshold(oof_probs, best_thresh)
    oof_f1_opt    = f1_score(oof_true, oof_preds_opt, average="macro", zero_division=0)
    print(f"  OOF Macro F1 (최적 threshold): {oof_f1_opt:.4f}")

    # ─────────────────────────────────────────────────────────
    # [STEP 16] Submission 생성 (두 가지 버전)
    # ─────────────────────────────────────────────────────────
    # 버전 A: 기본 argmax
    preds_raw  = test_probs.argmax(axis=1)
    labels_raw = [IDX2CLASS[i] for i in preds_raw]

    # 버전 B: 최적 threshold 적용
    preds_opt  = apply_threshold(test_probs, best_thresh)
    labels_opt = [IDX2CLASS[i] for i in preds_opt]

    # filename 순서를 sample_submission.csv 원본 순서 그대로 유지
    sub_raw = pd.DataFrame({"filename": test_filenames, "label": labels_raw})
    sub_opt = pd.DataFrame({"filename": test_filenames, "label": labels_opt})

    # sample_submission 순서로 정렬 보장
    order    = {fn: i for i, fn in enumerate(sample_df["filename"])}
    sub_raw  = sub_raw.sort_values("filename", key=lambda s: s.map(order)).reset_index(drop=True)
    sub_opt  = sub_opt.sort_values("filename", key=lambda s: s.map(order)).reset_index(drop=True)

    path_raw = os.path.join(CFG["output_dir"], "submission_raw.csv")
    path_opt = os.path.join(CFG["output_dir"], "submission_optimized.csv")

    sub_raw.to_csv(path_raw, index=False)
    sub_opt.to_csv(path_opt, index=False)

    print(f"\n{'═'*55}")
    print(f"  📄 제출 파일 생성 완료")
    print(f"  - {path_raw}  (기본)")
    print(f"  - {path_opt}  (threshold 최적화) ← 이걸 제출하세요")
    print(f"\n  예측 분포 (최적화 기준):")
    for cls, cnt in sub_opt["label"].value_counts().items():
        print(f"    {cls:<22}: {cnt:,}")
    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
