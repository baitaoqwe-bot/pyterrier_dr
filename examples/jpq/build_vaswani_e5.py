from pathlib import Path

import pyterrier as pt
import pyterrier_dr
import torch


def main() -> None:
    destination = Path("tests/fixtures/vaswani_e5.flex")

    if destination.exists():
        raise FileExistsError(
            f"Index already exists: {destination}\n"
            "Delete it manually only if you really want to rebuild it."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Destination: {destination}")

    e5 = pyterrier_dr.E5(device=device)

    index = pyterrier_dr.FlexIndex(str(destination))
    dataset = pt.get_dataset("vaswani")

    print("Building E5 Vaswani index...")

    pipeline = e5 >> index
    pipeline.index(dataset.get_corpus_iter())

    print("Index building completed.")
    print(f"Number of indexed documents: {len(index)}")


if __name__ == "__main__":
    main()