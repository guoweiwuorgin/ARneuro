"""Create the small PubMed-style Excel fixture used by the Toy example."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "input" / "pubmed_export.xlsx"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"PMID": "90000001", "Title": "Toy fMRI study of auditory word recognition", "Abstract": "A toy fMRI lexical decision study.", "DOI": "10.0000/toy.1", "PMCID": "", "Authors": "Example, A", "Journal/Book": "Journal of Toy Neuroscience", "Publication Year": "2024"},
            {"PMID": "90000002", "Title": "Toy EEG study of visuomotor attention", "Abstract": "A toy EEG target-detection study.", "DOI": "10.0000/toy.2", "PMCID": "", "Authors": "Example, B", "Journal/Book": "Toy Cognitive Science", "Publication Year": "2024"},
        ]
    ).to_excel(OUTPUT, index=False)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
