# Task 4: the Task 2 pipeline (pretrained MobileNet) with one change at a time.
# python experiments.py --variant 0 --seed 0          baseline
# python experiments.py --variant 1,2,8 --seed 0      several variants at once
# python experiments.py --variant 0 --seed 0 --export  also save the model as ONNX
#
# Variants:
#   0 baseline (= Task 2)
#   1 geometric augmentation     5 label smoothing 0.1
#   2 colour augmentation        6 AdamW, weight_decay 1e-2
#   3 frozen network             7 keep the best epoch on leaves_new/val
#   4 cosine LR, 30 epochs       8 + leaves_old/train
import argparse
import base64
import copy
import csv
import io
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--variant", default="0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--export", action="store_true")
args = parser.parse_args()
variants = [int(v) for v in args.variant.split(",")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| variants:", variants, "| seed:", args.seed)
os.makedirs("results", exist_ok=True)

# 2a. The data
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# The training transform: augmentation (variants 1, 2) only here, never for validation.
train_steps = []
if 1 in variants:
    train_steps += [
        transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
    ]
if 2 in variants:
    train_steps += [transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)]
train_transform = transforms.Compose(train_steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])

torch.manual_seed(args.seed)
train_ds = datasets.ImageFolder("data/leaves_new/train", transform=train_transform)
classes = train_ds.classes
if 8 in variants:
    old_train_ds = datasets.ImageFolder("data/leaves_old/train", transform=train_transform)
    assert old_train_ds.classes == classes
    train_ds = ConcatDataset([train_ds, old_train_ds])
new_val_ds = datasets.ImageFolder("data/leaves_new/val", transform=transform)
old_val_ds = datasets.ImageFolder("data/leaves_old/val", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
new_val_loader = DataLoader(new_val_ds, batch_size=32)
old_val_loader = DataLoader(old_val_ds, batch_size=32)
print(classes, "| train", len(train_ds), "| new val", len(new_val_ds), "| old val", len(old_val_ds))

# 2b. The network
model = models.mobilenet_v3_small(weights="DEFAULT")
if 3 in variants:
    # Frozen network: all pretrained parameters fixed, only the new last layer below is trained.
    for p in model.parameters():
        p.requires_grad = False
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model = model.to(device)

# 2c. Training
epochs = 30 if 4 in variants else 15
criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if 5 in variants else 0.0)
if 6 in variants:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
if 4 in variants:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def accuracy(loader):
    # Correct predictions / number of images, in evaluation mode without gradients.
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)


run_name = f"{args.variant} seed {args.seed}"
epoch_rows = []
best_acc, best_epoch, best_state = -1.0, epochs, None
start = time.time()
for epoch in range(epochs):
    model.train()
    loss_sum = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    if 4 in variants:
        scheduler.step()

    new_acc = accuracy(new_val_loader)
    old_acc = accuracy(old_val_loader)   # only recorded, never used for any choice
    print(f"epoch {epoch + 1:2d}  training loss {loss_sum / len(train_loader):.4f}  "
          f"new val {new_acc:.3f}  old val {old_acc:.3f}")
    epoch_rows.append([args.variant, args.seed, epoch + 1, loss_sum / len(train_loader), new_acc, old_acc])

    if 7 in variants and new_acc > best_acc:
        # The best epoch is chosen on leaves_new/val only.
        best_acc, best_epoch = new_acc, epoch + 1
        best_state = copy.deepcopy(model.state_dict())
train_time = time.time() - start

if 7 in variants:
    model.load_state_dict(best_state)
final_new_acc = accuracy(new_val_loader)
final_old_acc = accuracy(old_val_loader)
print(f"final ({run_name}, model of epoch {best_epoch}): new val {final_new_acc:.3f}  "
      f"old val {final_old_acc:.3f}  time {train_time:.0f} s")

# Results: one row per run, one row per epoch (rows are appended).
new_file = not os.path.exists("results/results.csv")
with open("results/results.csv", "a", newline="") as f:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["variant", "seed", "acc_new_val", "acc_old_val", "best_epoch", "epochs", "train_time_s"])
    writer.writerow([args.variant, args.seed, f"{final_new_acc:.4f}", f"{final_old_acc:.4f}",
                     best_epoch, epochs, f"{train_time:.1f}"])

new_file = not os.path.exists("results/epochs.csv")
with open("results/epochs.csv", "a", newline="") as f:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["variant", "seed", "epoch", "train_loss", "acc_new_val", "acc_old_val"])
    for row in epoch_rows:
        writer.writerow(row[:3] + [f"{x:.4f}" for x in row[3:]])

# results/report.html, generated from the two CSV files.
with open("results/results.csv", newline="") as f:
    result_rows = list(csv.reader(f))
with open("results/epochs.csv", newline="") as f:
    epoch_table = list(csv.DictReader(f))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for run in dict.fromkeys((r["variant"], r["seed"]) for r in epoch_table):
    rows = [r for r in epoch_table if (r["variant"], r["seed"]) == run]
    x = [int(r["epoch"]) for r in rows]
    for ax, column in zip(axes, ["acc_new_val", "acc_old_val"]):
        ax.plot(x, [float(r[column]) for r in rows], label=f"variant {run[0]} seed {run[1]}")
for ax, title in zip(axes, ["leaves_new/val", "leaves_old/val (main metric)"]):
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("accuracy")
axes[1].legend(fontsize=8, loc="lower right")
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
plt.close(fig)
image = base64.b64encode(buffer.getvalue()).decode()

table = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in result_rows[1:])
header = "".join(f"<th>{cell}</th>" for cell in result_rows[0])
with open("results/report.html", "w", encoding="utf-8") as f:
    f.write(f"""<!doctype html><html><head><meta charset="utf-8"><title>Leaf experiments</title>
<style>body {{font-family: sans-serif; margin: 20px}} td, th {{border: 1px solid #ccc; padding: 4px 8px; text-align: right}}
table {{border-collapse: collapse}} img {{max-width: 100%}}</style></head><body>
<h1>Leaf experiments</h1>
<table><tr>{header}</tr>{table}</table>
<h2>Accuracy per epoch</h2>
<img src="data:image/png;base64,{image}">
</body></html>""")
print("results/results.csv, results/epochs.csv, results/report.html updated")

# 2e. The model as a file
if args.export:
    import onnx

    onnx_file = f"results/leaf_model_v{args.variant.replace(',', '+')}_s{args.seed}.onnx"
    model.eval().cpu()
    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), onnx_file,
                      input_names=["image"], output_names=["logits"], external_data=False)

    model_info = {"classes": classes, "input_size": 96, "mean": mean, "std": std}
    onnx_model = onnx.load(onnx_file)
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, onnx_file)
    print(f"{onnx_file}: {os.path.getsize(onnx_file) / 1e6:.1f} MB")
