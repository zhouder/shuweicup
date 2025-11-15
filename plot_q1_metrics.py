import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== 基本配置 ======
CKPT_DIR = "checkpoints(zhouder)"
CSV_PATH = os.path.join(CKPT_DIR, "task1_metrics.csv")

OUT_DIR = "figures_q1"   # 保存图片的目录
os.makedirs(OUT_DIR, exist_ok=True)

# ====== 读取数据 ======
df = pd.read_csv(CSV_PATH)

# 拆分 train / val
train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()

# 为了保险，按 epoch 排一下序
train_df = train_df.sort_values("epoch")
val_df   = val_df.sort_values("epoch")

# ====== 1. Loss 曲线 ======
plt.figure()
plt.plot(train_df["epoch"], train_df["loss"], label="train")
plt.plot(val_df["epoch"],   val_df["loss"],   label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Q1 Train / Val Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q1_loss_curve.png"), dpi=300)
plt.close()

# ====== 2. Top-1 Accuracy 曲线（acc1） ======
plt.figure()
plt.plot(train_df["epoch"], train_df["acc1"], label="train")
plt.plot(val_df["epoch"],   val_df["acc1"],   label="val")
plt.xlabel("Epoch")
plt.ylabel("Top-1 Accuracy (acc1)")
plt.title("Q1 Train / Val Top-1 Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q1_acc1_curve.png"), dpi=300)
plt.close()

# ====== 3. Top-3 Accuracy 曲线（acc3） ======
plt.figure()
plt.plot(train_df["epoch"], train_df["acc3"], label="train")
plt.plot(val_df["epoch"],   val_df["acc3"],   label="val")
plt.xlabel("Epoch")
plt.ylabel("Top-3 Accuracy (acc3)")
plt.title("Q1 Train / Val Top-3 Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q1_acc3_curve.png"), dpi=300)
plt.close()

# ====== 4. Macro-F1 曲线 ======
plt.figure()
plt.plot(train_df["epoch"], train_df["macro_f1"], label="train")
plt.plot(val_df["epoch"],   val_df["macro_f1"],   label="val")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Q1 Train / Val Macro-F1")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q1_macro_f1_curve.png"), dpi=300)
plt.close()

# ====== 5. （可选）Macro-Precision / Macro-Recall 曲线 ======
plt.figure()
plt.plot(train_df["epoch"], train_df["macro_precision"], label="train_precision")
plt.plot(val_df["epoch"],   val_df["macro_precision"],   label="val_precision")
plt.xlabel("Epoch")
plt.ylabel("Macro Precision")
plt.title("Q1 Train / Val Macro-Precision")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q1_macro_precision_curve.png"), dpi=300)
plt.close()

plt.figure()
plt.plot(train_df["epoch"], train_df["macro_recall"], label="train_recall")
plt.plot(val_df["epoch"],   val_df["macro_recall"],   label="val_recall")
plt.xlabel("Epoch")
plt.ylabel("Macro Recall")
plt.title("Q1 Train / Val Macro-Recall")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "q1_macro_recall_curve.png"), dpi=300)
plt.close()

print("Done. Figures saved in:", OUT_DIR)
