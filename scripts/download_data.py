"""
Download the three datasets used in the paper.

  * BATADAL      -> data/batadal/  (dataset03 + dataset04, from the official
                    batadal.net GitHub mirror)
  * Gas Pipeline -> data/gas/IanArffDataset.arff  (Morris/Turnipseed ARFF)
  * HAI 22.04    -> data/hai/  (train1 + test1 from the official repository)

Usage:
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset batadal
    python scripts/download_data.py --dataset gas_pipeline
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]

BATADAL = {
    "BATADAL_dataset03.csv":
        "https://raw.githubusercontent.com/scy-phy/www.batadal.net/master/data/BATADAL_dataset03.csv",
    "BATADAL_dataset04.csv":
        "https://raw.githubusercontent.com/scy-phy/www.batadal.net/master/data/BATADAL_dataset04.csv",
}
GAS_URL = "https://raw.githubusercontent.com/Rocionightwater/ML-NIDS-for-SCADA/master/data/IanArffDataset.arff"
HAI_BASE = "https://media.githubusercontent.com/media/icsdataset/hai/master/hai-22.04"
HAI_FILES = ["train1.csv", "test1.csv"]   # complete files used in the paper


def _get(url, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"  [skip] {out.name} already present ({out.stat().st_size:,} B)")
        return
    print(f"  downloading {out.name} ...")
    urlretrieve(url, out)
    print(f"  [ok] {out}  ({out.stat().st_size:,} B)")


def batadal():
    print("[BATADAL]")
    for name, url in BATADAL.items():
        _get(url, ROOT / "data" / "batadal" / name)


def gas():
    print("[Gas Pipeline]")
    _get(GAS_URL, ROOT / "data" / "gas" / "IanArffDataset.arff")


def hai():
    print("[HAI 22.04 subset]")
    for f in HAI_FILES:
        _get(f"{HAI_BASE}/{f}", ROOT / "data" / "hai" / f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["all", "batadal", "gas_pipeline", "hai"], default="all")
    a = p.parse_args()
    if a.dataset in ("all", "batadal"):
        batadal()
    if a.dataset in ("all", "gas_pipeline"):
        gas()
    if a.dataset in ("all", "hai"):
        hai()
    print("\nDone. Verify with: python scripts/verify_results.py --check-data")


if __name__ == "__main__":
    main()

