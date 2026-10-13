"""Download the pinned LDM checkpoint into models/ldm-text2im-large-256/.

    python -m scripts.download_model

Skips files that already exist with the expected size, and checks the
SHA-256 of every large file it downloads against models/model_manifest.json.
"""
import hashlib
import json

from huggingface_hub import hf_hub_download

from utils.config import MANIFEST_PATH, MODEL_DIR, MODEL_FILES, MODEL_ID, REVISION


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = {f["rfilename"]: f for f in json.loads(MANIFEST_PATH.read_text())["files"]}
    for name in MODEL_FILES:
        entry = manifest[name]
        path = MODEL_DIR / name
        if path.is_file() and path.stat().st_size == entry["size"]:
            print(f"ok        {name}")
            continue
        print(f"download  {name} ({entry['size'] / 2**20:.1f} MiB)")
        hf_hub_download(MODEL_ID, name, revision=REVISION, local_dir=MODEL_DIR)
        if "lfs" in entry and sha256(path) != entry["lfs"]["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {name}; delete it and run again.")
    print(f"All files are in {MODEL_DIR}")


if __name__ == "__main__":
    main()
