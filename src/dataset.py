from __future__ import annotations

import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_CLASS_ORDER = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "Scratches",
]

CLASS_ALIASES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted_Surface",
    "pitted-surface": "Pitted_Surface",
    "rolled-in_scale": "Rolled-in_Scale",
    "rolled_in_scale": "Rolled-in_Scale",
    "rolled-in-scale": "Rolled-in_Scale",
    "scratches": "Scratches",
}

FILENAME_LABEL_PATTERN = re.compile(r"([A-Za-z_-]+)_\d+", re.IGNORECASE)


@dataclass
class SplitData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    input_dim: int
    resolved_data_dir: str


class FlatImageDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: Callable | None = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FlattenTransform:
    def __init__(self, img_size: int, augment: bool = False):
        self.img_size = img_size
        self.augment = augment

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.img_size, self.img_size))

        if self.augment:
            angle = random.uniform(-8, 8)
            image = image.rotate(angle, fillcolor=0)

            dx = int(random.uniform(-0.05, 0.05) * self.img_size)
            dy = int(random.uniform(-0.05, 0.05) * self.img_size)
            canvas = Image.new("L", (self.img_size, self.img_size), color=0)
            canvas.paste(image, (dx, dy))
            image = canvas

            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)

        pixels = list(image.getdata())
        tensor = torch.tensor(pixels, dtype=torch.float32) / 255.0
        return tensor.view(-1)


def _normalize_label_name(name: str) -> str | None:
    key = name.strip().lower().replace(" ", "_")
    return CLASS_ALIASES.get(key)


def _ordered_class_names(names: list[str]) -> list[str]:
    unique = sorted(set(names))
    if set(unique).issubset(set(DEFAULT_CLASS_ORDER)):
        return [name for name in DEFAULT_CLASS_ORDER if name in unique]
    return unique


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []

    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            result.append(p)

    return result


def _find_existing_data_path(data_path: Path) -> Path:
    project_root = Path(__file__).resolve().parent.parent

    candidates: list[Path] = [data_path]

    if not data_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / data_path,
                project_root / data_path,
                project_root / data_path.name,
            ]
        )

    candidates.extend(
        [
            project_root / "NEU-CLS.zip",
            project_root / "NEU-CLS",
            project_root / "data" / data_path.name,
            project_root / "data" / "NEU-CLS.zip",
            project_root / "data" / "NEU-CLS",
        ]
    )

    for candidate in _unique_paths(candidates):
        if candidate.exists():
            return candidate

    tried = "\n- " + "\n- ".join(str(p) for p in _unique_paths(candidates))
    raise FileNotFoundError(
        "Không tìm thấy dữ liệu. Đã thử các vị trí sau:" + tried
    )


def _extract_zip_if_needed(data_path: Path) -> Path:
    resolved_path = _find_existing_data_path(data_path)

    if resolved_path.is_dir():
        return resolved_path

    if resolved_path.is_file() and resolved_path.suffix.lower() == ".zip":
        extract_root = resolved_path.parent / f"{resolved_path.stem}_extracted"
        marker = extract_root / ".extracted_ok"

        if not marker.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(resolved_path, "r") as zf:
                zf.extractall(extract_root)
            marker.write_text("ok", encoding="utf-8")

        return extract_root

    raise FileNotFoundError(f"Không tìm thấy dữ liệu tại: {resolved_path}")


def _iter_candidate_roots(root: Path):
    seen: set[str] = set()

    def _yield_if_new(p: Path):
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen and p.is_dir():
            seen.add(key)
            return p
        return None

    first = _yield_if_new(root)
    if first is not None:
        yield first

    level1 = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    for p in level1:
        item = _yield_if_new(p)
        if item is not None:
            yield item

    for parent in level1:
        level2 = [p for p in sorted(parent.iterdir()) if p.is_dir() and not p.name.startswith(".")]
        for p in level2:
            item = _yield_if_new(p)
            if item is not None:
                yield item


def _scan_class_folders(root: Path) -> tuple[list[tuple[Path, int]], list[str]] | None:
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not class_dirs:
        return None

    normalized: list[tuple[Path, str]] = []
    for class_dir in class_dirs:
        class_name = _normalize_label_name(class_dir.name)
        if class_name is None:
            return None
        normalized.append((class_dir, class_name))

    class_names = _ordered_class_names([name for _, name in normalized])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples: list[tuple[Path, int]] = []
    for class_dir, class_name in normalized:
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                samples.append((path, class_to_idx[class_name]))

    return (samples, class_names) if samples else None


def _infer_label_from_filename(path: Path) -> str | None:
    match = FILENAME_LABEL_PATTERN.match(path.stem)
    if not match:
        return None
    return _normalize_label_name(match.group(1))


def _scan_flat_images(root: Path) -> tuple[list[tuple[Path, int]], list[str]] | None:
    image_paths = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]
    if not image_paths:
        return None

    labels_by_path: list[tuple[Path, str]] = []
    for path in image_paths:
        class_name = _infer_label_from_filename(path)
        if class_name is not None:
            labels_by_path.append((path, class_name))

    if not labels_by_path:
        return None

    class_names = _ordered_class_names([class_name for _, class_name in labels_by_path])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples = [(path, class_to_idx[class_name]) for path, class_name in labels_by_path]
    return samples, class_names


def _resolve_samples(data_dir: str | Path) -> tuple[list[tuple[Path, int]], list[str], Path]:
    extracted_root = _extract_zip_if_needed(Path(data_dir))
    candidate_roots = list(_iter_candidate_roots(extracted_root))

    for candidate_root in candidate_roots:
        direct_scan = _scan_class_folders(candidate_root)
        if direct_scan is not None:
            samples, class_names = direct_scan
            return samples, class_names, candidate_root

    for candidate_root in candidate_roots:
        flat_scan = _scan_flat_images(candidate_root)
        if flat_scan is not None:
            samples, class_names = flat_scan
            return samples, class_names, candidate_root

    raise ValueError(
        "Không đọc được dữ liệu. Hỗ trợ 2 kiểu:\n"
        "(1) thư mục lớp như Crazing/, Inclusion/, ...\n"
        "(2) thư mục phẳng hoặc .zip có tên ảnh dạng crazing_10.jpg, inclusion_20.jpg, ..."
    )


def create_dataloaders(
    data_dir: str | Path = "NEU-CLS.zip",
    img_size: int = 64,
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    augment: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> SplitData:
    samples, class_names, resolved_root = _resolve_samples(data_dir)

    paths = [str(p) for p, _ in samples]
    labels = [label for _, label in samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=random_state,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=random_state,
    )

    train_samples = [(Path(p), y) for p, y in zip(train_paths, train_labels)]
    val_samples = [(Path(p), y) for p, y in zip(val_paths, val_labels)]
    test_samples = [(Path(p), y) for p, y in zip(test_paths, test_labels)]

    train_ds = FlatImageDataset(
        train_samples,
        transform=FlattenTransform(img_size, augment=augment),
    )
    eval_tf = FlattenTransform(img_size, augment=False)
    val_ds = FlatImageDataset(val_samples, transform=eval_tf)
    test_ds = FlatImageDataset(test_samples, transform=eval_tf)

    return SplitData(
        train_loader=DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        val_loader=DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        test_loader=DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        class_names=class_names,
        input_dim=img_size * img_size,
        resolved_data_dir=str(resolved_root),
    )