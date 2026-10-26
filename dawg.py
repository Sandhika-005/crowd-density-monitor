from pathlib import Path

sample_dir = Path("Dataset/part_B/sample")
images_dir = sample_dir / "images"
labels_dir = sample_dir / "ground-truth"

keep_images = {f"IMG_{i}.jpg" for i in range(1, 11)}
keep_labels = {f"GT_IMG_{i}.mat" for i in range(1, 11)}

for folder, keep in [(images_dir, keep_images), (labels_dir, keep_labels)]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")

    for file in folder.iterdir():
        if file.is_file() and file.name not in keep:
            file.unlink()
            print("Dihapus:", file)

print("Selesai. Hanya IMG_1–10.jpg dan GT_IMG_1–10.mat yang disisakan.")