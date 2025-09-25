# CNNclassify.py
# Runs a custom model for training, testing and a pretrained resnet-20 using the following commands:
#   - python CNNclassify.py train
#   - python CNNclassify.py test <image_path>
#   - python CNNclassify.py resnet20
from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from resnet20_cifar import ResNet

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class CustomCNN(nn.Module):
    """Custom CNN for CIFAR-10.
    First conv layer is fixed by spec:
    - kernel_size=5, stride=1, padding=0, out_channels=32.
    """

    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Required first layer (do not change)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def plot_accuracies(
    train_accs, test_accs, epochs, save_path=f"./model/accuracy_curve_seed-{seed}.png"
):
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, epochs + 1),
        train_accs,
        label="Train Accuracy",
        color="blue",
        marker="o",
    )
    plt.plot(
        range(1, epochs + 1), test_accs, label="Test Accuracy", color="red", marker="o"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training vs Testing Accuracy")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Accuracy curve saved at {save_path}")


def get_test_transform():
    """Apply a 32x32 resize and Normalize: transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))"""
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def get_cifar10_test_loader(batch_size: int = 128, data_root: str = "./data"):
    test_transform = get_test_transform()
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return test_loader


def get_cifar10_train_loader(batch_size: int = 128, data_root: str = "./data"):
    """Return CIFAR-10 train/test loaders with standard transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.2),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader


def _print_metrics_header():
    print(
        f"{'Loop':>8}  {'Train Loss':>15}  {'Train Acc %':>15}  {'Test Loss':>15}  {'Test Acc %':>15}"
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Return (avg_loss, avg_acc_percent) over the loader."""
    model.eval()
    running_loss, running_correct, running_count = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * targets.size(0)
        running_correct += (logits.argmax(1) == targets).sum().item()
        running_count += targets.size(0)

    avg_loss = running_loss / running_count
    avg_acc = 100.0 * running_correct / running_count
    return avg_loss, avg_acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    device: torch.device,
) -> tuple[float, float]:
    """Runs one training epoch and returns (avg_loss, avg_acc_percent)."""
    model.train()
    running_loss, running_correct, running_count = 0.0, 0, 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * targets.size(0)
        running_correct += (logits.argmax(1) == targets).sum().item()
        running_count += targets.size(0)

    avg_loss = running_loss / running_count
    avg_acc = 100.0 * running_correct / running_count
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_pt_resnet_20_cifar10(
    model: nn.Module,
    batch_size: int = 128,
    data_root: str = "./data",
    device: torch.device | None = None,
) -> float:
    """Evalaute a pt resent 20 model"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_cifar10_test_loader(batch_size=batch_size, data_root=data_root)
    model.eval()
    correct, count = 0, 0
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        correct += (logits.argmax(1) == targets).sum().item()
        count += targets.size(0)
    return 100.0 * correct / count


def resnet20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/resnet20_cifar10.pt"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = ResNet(depth=20, num_classes=10).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    acc = evaluate_pt_resnet_20_cifar10(
        model, batch_size=128, data_root="./data", device=device
    )
    print(f"ResNet-20 CIFAR-10 test accuracy: {acc:.2f}%")


def test(image: str):
    """Run inference on a single given image never a batch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/custom_cnn.pt"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = CustomCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Hook to capture conv1 feature maps
    conv1_out = {}

    def _hook(_m, _i, o):
        conv1_out["x"] = o.detach().cpu()  # [1, 32, 28, 28]

    first_conv = model.conv_layers[0]
    handle = first_conv.register_forward_hook(_hook)

    # Prepare image (RGB, 32x32, CIFAR-10 normalization)
    img = Image.open(image).convert("RGB")
    x = get_test_transform()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
    handle.remove()

    # Print predicted class (exactly one line)
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    pred_idx = int(logits.argmax(dim=1).item())
    print(classes[pred_idx])

    # Save conv1 visualization as CONV_rslt.png (32 maps in an 8x4 grid, grayscale)
    feats = conv1_out["x"][0]  # [32, 28, 28]
    fmin = feats.amin(dim=(1, 2), keepdim=True)
    fmax = feats.amax(dim=(1, 2), keepdim=True)
    feats_norm = (feats - fmin) / (fmax - fmin + 1e-6)  # per-map min-max
    grid = vutils.make_grid(feats_norm.unsqueeze(1), nrow=8, padding=1)
    vutils.save_image(grid, "CONV_rslt.png")


def train():
    """Train CustomCNN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 128
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    train_loader = get_cifar10_train_loader(batch_size=batch_size, data_root="./data")
    test_loader = get_cifar10_test_loader(batch_size=batch_size, data_root="./data")
    model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.3,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        div_factor=10.0,  # initial lr = max_lr/div_factor
        final_div_factor=1e3,  # anneal close to 0
    )

    train_accs, test_accs = [], []
    _print_metrics_header()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        te_loss, te_acc = evaluate_one_epoch(model, test_loader, criterion, device)

        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        # Loop, Train Loss, Train Acc %, Test Loss, Test Acc %
        print(
            f"{epoch:>4}/{epochs:<3}  {tr_loss:>15.4f}  {tr_acc:>15.4f}  {te_loss:>15.4f}  {te_acc:>15.4f}"
        )

    os.makedirs("./model", exist_ok=True)
    torch.save(model.state_dict(), "./model/custom_cnn.pt")
    plot_accuracies(train_accs, test_accs, epochs)


def main():
    parser = ArgumentParser(
        prog="CNNclassify.py", description="CNN classifier for CIFAR-10"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    subparsers.add_parser("train", help="Run in training mode")

    # Test subcommand
    test_parser = subparsers.add_parser(
        "test", help="Run in testing mode with an image file"
    )
    test_parser.add_argument("image", type=str, help="Path to image file (.png)")

    # ResNet20 subcommand
    subparsers.add_parser("resnet20", help="Run in ResNet20 mode")

    args = parser.parse_args()

    if args.command == "train":
        train()

    elif args.command == "test":
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"Test file not found: {args.image}")
        if not args.image.lower().endswith(".png"):
            raise ValueError("Test file must be a (.png) type image")
        test(args.image)

    elif args.command == "resnet20":
        resnet20()


if __name__ == "__main__":
    main()
