#!/usr/bin/env python3
"""Reassemble data/raw/high_freq_data.csv bit-for-bit from <=25 MB standalone .zip tomes
and verify exact SHA-256 checksum against the original 256.84 MB dataset."""

from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "raw" / "high_freq_data_tomes" / "tomes_manifest.json"

def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tomes_dir = ROOT / manifest["tomes_dir"]
    target = ROOT / manifest["target_file"]
    target.parent.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha256()
    total_rows = 0
    with target.open("wb") as out_f:
        for idx, rec in enumerate(manifest["tomes"]):
            zp = tomes_dir / rec["zip_file"]
            if not zp.is_file():
                raise FileNotFoundError(f"Missing tome: {zp}")
            with zipfile.ZipFile(zp, "r") as zf:
                with zf.open(rec["inner_csv"], "r") as member:
                    hdr = member.readline()
                    if idx == 0:
                        out_f.write(hdr)
                        h.update(hdr)
                    while True:
                        chunk = member.read(1 << 20)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        h.update(chunk)
            total_rows += int(rec["data_rows"])
            print(f"  Merged {rec['zip_file']} ({rec['data_rows']:,} rows)")

    got_sha = h.hexdigest()
    exp_sha = manifest["original_sha256"]
    if got_sha != exp_sha:
        raise RuntimeError(f"SHA-256 mismatch! expected={exp_sha}, got={got_sha}")
    print(f"\nSUCCESS: Restored {target} ({target.stat().st_size/1048576:.2f} MB, {total_rows:,} rows)")
    print(f"SHA-256 verified bit-for-bit: {got_sha}")

if __name__ == "__main__":
    main()
