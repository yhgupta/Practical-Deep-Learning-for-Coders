import argparse
from io import BytesIO
from pathlib import Path

import requests
import vtracer
from PIL import Image


def download_image(url: str, out_path: Path) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    img.save(out_path)


def raster_to_svg(input_path: Path, output_path: Path) -> None:
    vtracer.convert_image_to_svg_py(
        str(input_path),
        str(output_path),
        colormode="color",
        mode="spline",
        filter_speckle=4,
        color_precision=6,
        corner_threshold=60,
        length_threshold=4,
    )


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Image URL (jpg/png)")
    src.add_argument("--input", help="Local image path")
    parser.add_argument("--name", default="sample", help="Output base name")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    raster_path = out_dir / f"{args.name}.png"
    svg_path = out_dir / f"{args.name}.svg"

    if args.url:
        download_image(args.url, raster_path)
    else:
        img = Image.open(args.input).convert("RGB")
        img.save(raster_path)

    raster_to_svg(raster_path, svg_path)

    print(f"Saved raster: {raster_path}")
    print(f"Saved svg   : {svg_path}")

if __name__ == "__main__":
    main()