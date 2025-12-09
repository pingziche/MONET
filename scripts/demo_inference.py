import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

DEFAULT_LABELS = [
    "acne",
    "eczema",
    "psoriasis",
    "basal cell carcinoma",
    "squamous cell carcinoma",
    "melanoma",
    "nevus",
    "urticaria",
    "rosacea",
    "impetigo",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple MONET zero-shot demo on an image.")
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=Path("test.jpg"),
        help="Path to the input image. Defaults to test.jpg in the repository root.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated candidate labels for zero-shot classification.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top-scoring labels to display.",
    )
    return parser.parse_args()


def load_model(device: torch.device):
    processor = AutoProcessor.from_pretrained("chanwkim/monet")
    model = AutoModelForZeroShotImageClassification.from_pretrained("chanwkim/monet")
    model.to(device)
    model.eval()
    return model, processor


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image at {image_path.resolve()}")
    return Image.open(image_path).convert("RGB")


def run_inference(image_path: Path, candidate_labels: list[str], top_k: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(device)

    image = load_image(image_path)
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probabilities = logits.softmax(dim=0)

    top_k = min(top_k, len(candidate_labels))
    scores, indices = torch.topk(probabilities, k=top_k)

    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print("\nTop predictions:")
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        label = candidate_labels[idx]
        print(f"{rank}. {label:<25} {score.item():.4f}")


def main():
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("At least one label must be provided.")
    run_inference(args.image, labels, args.top_k)


if __name__ == "__main__":
    main()
