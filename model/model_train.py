# train_and_export_mnist_q8_24.py
# Trains a simple MNIST MLP (784->128->10), quantizes to Q8.24, and writes HEX files.
# Requires: torch, torchvision
# Run: python train_and_export_mnist_q8_24.py --epochs 5 --hidden 128 --export_images 10

import argparse, json, os, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ----------------------------
# Q8.24 helpers (signed 32-bit)
# ----------------------------
FRAC_BITS = 24
SCALE = 1 << FRAC_BITS
INT32_MIN = - (1 << 31)
INT32_MAX =   (1 << 31) - 1

def float_to_q8_24(x: torch.Tensor) -> torch.Tensor:
    """
    Convert float tensor to Q8.24, with rounding and saturation to signed int32.
    Returns int32 tensor.
    """
    # Round-to-nearest
    y = torch.round(x * SCALE)
    # Saturate
    y = torch.clamp(y, INT32_MIN, INT32_MAX)
    return y.to(torch.int64)  # use int64 for safe export; still 32-bit range

def int_to_hex32(val: int) -> str:
    """
    Convert signed int32 value to 8-hex-digit two's complement uppercase string (no 0x).
    """
    val &= 0xFFFFFFFF
    return f"{val:08X}"

def write_hex_per_line(vec, path: Path):
    """
    vec: 1D iterable of Python ints in int32 range.
    Writes one 8-hex-digit word per line (two's complement).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in vec:
            f.write(int_to_hex32(int(v)) + "\n")

# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, x):
        # x: [N,1,28,28] -> flatten to [N,784]
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ----------------------------
# Training / Eval
# ----------------------------
def train_one_epoch(model, loader, criterion, opt, device):
    model.train()
    running = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
    return loss_sum / total, correct / total

# ----------------------------
# Exporters
# ----------------------------
def export_weights_biases_q8_24(model: MLP, outdir: Path):
    """
    Exports:
      - W1.hex (OUT=hidden, IN=784; order: for out in 0..OUT-1: for in in 0..IN-1)
      - B1.hex (hidden)
      - W2.hex (OUT=10, IN=hidden)
      - B2.hex (10)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Get float32 weights/biases
    W1_f = model.fc1.weight.detach().cpu().clone()   # [hidden, 784]
    B1_f = model.fc1.bias.detach().cpu().clone()     # [hidden]
    W2_f = model.fc2.weight.detach().cpu().clone()   # [10, hidden]
    B2_f = model.fc2.bias.detach().cpu().clone()     # [10]

    # Quantize to Q8.24 signed int
    W1_q = float_to_q8_24(W1_f)
    B1_q = float_to_q8_24(B1_f)
    W2_q = float_to_q8_24(W2_f)
    B2_q = float_to_q8_24(B2_f)

    # Flatten in (OUT major, then IN) order
    W1_flat = W1_q.reshape(-1).tolist()  # already [OUT, IN] row-major
    W2_flat = W2_q.reshape(-1).tolist()

    # Write hex files
    write_hex_per_line(W1_flat, outdir / "W1.hex")
    write_hex_per_line(B1_q.tolist(), outdir / "B1.hex")
    write_hex_per_line(W2_flat, outdir / "W2.hex")
    write_hex_per_line(B2_q.tolist(), outdir / "B2.hex")

def export_images_q8_24(loader, outdir: Path, count: int = 10):
    """
    Exports first 'count' images from 'loader' as Q8.24 hex vectors of length 784.
    Files: image_0000.hex, image_0001.hex, ...
    Also writes labels.json with the ground-truth labels for those images.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    labels_out = []
    for images, labels in loader:
        for k in range(images.size(0)):
            if saved >= count: 
                with open(outdir / "labels.json", "w") as f:
                    json.dump(labels_out, f, indent=2)
                return
            img = images[k].view(-1).cpu()  # [784], floats in [0,1]
            img_q = float_to_q8_24(img)     # int Q8.24
            write_hex_per_line(img_q.tolist(), outdir / f"image_{saved:04d}.hex")
            labels_out.append(int(labels[k].item()))
            saved += 1
    # If dataset smaller than count
    with open(outdir / "labels.json", "w") as f:
        json.dump(labels_out, f, indent=2)

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--export_dir", type=str, default="mnist_q8_24_export")
    p.add_argument("--export_images", type=int, default=10, help="How many test images to export as hex (0 to skip)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data: normalize to [0,1] (ToTensor does that already)
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True if device.type=="cuda" else False)
    test_ld  = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    model = MLP(in_dim=784, hidden=args.hidden, out_dim=10).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device} for {args.epochs} epochs...")
    for e in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_ld, criterion, opt, device)
        te_loss, te_acc = evaluate(model, test_ld, device)
        print(f"Epoch {e:02d}: train_loss={tr_loss:.4f} | test_loss={te_loss:.4f} | test_acc={te_acc*100:.2f}%")

    # Final eval
    test_loss, test_acc = evaluate(model, test_ld, device)
    print(f"Final: test_acc={test_acc*100:.2f}%")

    outdir = Path(args.export_dir)
    export_weights_biases_q8_24(model, outdir)

    if args.export_images > 0:
        export_images_q8_24(test_ld, outdir, count=args.export_images)

    meta = {
        "in_dim": 784,
        "hidden": args.hidden,
        "out_dim": 10,
        "q_format": "Q8.24 (signed)",
        "frac_bits": FRAC_BITS,
        "scale": SCALE,
        "test_acc": test_acc,
        "notes": [
            "W1.hex order: for out in [0..hidden-1], for in in [0..783], one 32-bit HEX word per line.",
            "B1.hex: hidden lines; W2.hex: for out in [0..9], for in in [0..hidden-1].",
            "B2.hex: 10 lines.",
            "image_xxxx.hex: 784 lines (flattened row-major 28x28) in Q8.24.",
            "Two's complement 32-bit (8 hex chars per line, uppercase)."
        ]
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported to: {outdir.resolve()}")
    print("Files: W1.hex, B1.hex, W2.hex, B2.hex, (optional) image_*.hex, meta.json")

if __name__ == "__main__":
    main()
