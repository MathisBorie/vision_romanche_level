"""Accès au bucket des photos de la Romanche (AWS S3 ou Cloudflare R2).

La configuration est lue dans le fichier .env à la racine du projet (voir .env.example) :
    S3_BUCKET, S3_PREFIX, S3_ENDPOINT_URL, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

Utilisation :
    from bucket import Bucket
    b = Bucket()
    images = b.list_images(since_hours=24)
    paths = b.download(images)
    img = b.load_image(images[-1]["Key"])
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"  # data/ est ignoré par git
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

load_dotenv(ROOT / ".env")


class Bucket:
    def __init__(self, bucket: str | None = None, prefix: str | None = None):
        self.name = bucket or os.environ["S3_BUCKET"]
        self.prefix = prefix if prefix is not None else os.getenv("S3_PREFIX", "")
        endpoint = os.getenv("S3_ENDPOINT_URL") or None  # vide = AWS
        region = os.getenv("AWS_REGION") or ("auto" if endpoint else "eu-west-3")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        self.endpoint = endpoint

    def __repr__(self) -> str:
        return f"Bucket({self.name!r}, prefix={self.prefix!r}, endpoint={self.endpoint or 'AWS'})"

    def list_images(self, since: datetime | None = None, since_hours: float | None = None,
                    prefix: str | None = None) -> list[dict]:
        """Objets image (dicts boto3 : Key, LastModified, Size...), du plus ancien au plus récent."""
        if since_hours is not None:
            since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        prefix = self.prefix if prefix is None else prefix
        objs = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.name, Prefix=prefix):
            for o in page.get("Contents", []):
                if o["Key"].lower().endswith(IMG_EXT) and (since is None or o["LastModified"] >= since):
                    objs.append(o)
        return sorted(objs, key=lambda o: o["LastModified"])

    def ls(self, prefix: str = "") -> tuple[list[str], list[dict]]:
        """Contenu d'un 'dossier' du bucket : (sous-dossiers, images). Ne descend pas plus bas."""
        folders, files = [], []
        for page in self.s3.get_paginator("list_objects_v2").paginate(
                Bucket=self.name, Prefix=prefix, Delimiter="/"):
            folders += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
            files += [o for o in page.get("Contents", []) if o["Key"].lower().endswith(IMG_EXT)]
        return folders, files

    def latest(self, n: int = 1) -> list[dict]:
        """Les n images les plus récentes (liste tout le préfixe)."""
        return self.list_images()[-n:]

    def download(self, objs: list[dict], dest: Path = DATA_DIR) -> list[Path]:
        """Télécharge dans dest en gardant l'arborescence des clés ; saute les fichiers déjà présents."""
        paths = []
        for o in objs:
            p = Path(dest) / o["Key"]
            if not (p.exists() and p.stat().st_size == o["Size"]):
                p.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.name, o["Key"], str(p))
            paths.append(p)
        return paths

    def read_bytes(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.name, Key=key)["Body"].read()

    def load_image(self, key: str):
        """Image PIL lue directement en mémoire, sans écrire sur disque."""
        from PIL import Image
        return Image.open(io.BytesIO(self.read_bytes(key)))


if __name__ == "__main__":
    b = Bucket()
    imgs = b.list_images(since_hours=24)
    print(b, f"\n{len(imgs)} images ces dernières 24 h")
    for o in imgs[-5:]:
        print(o["LastModified"].isoformat(), o["Key"])
