"""Concatenate paired ranklist images (baseline on top, ours on bottom).

For every image name that exists in both directories, stack the top image
above the bottom image with a small vertical gap, and save the result.

Usage:
    python concat_ranklist.py \
        --top    /xxx/extrawork/code/svc_net/ranklist/market/baseline \
        --bottom /xxx/extrawork/code/svc_net/ranklist/market/market_01_oval_single \
        --output /xxx/extrawork/code/svc_net/ranklist/market/compare_baseline_vs_oval

Notes:
    - Only files with the same filename in both dirs are concatenated.
    - If widths differ, the narrower image is resized to match the wider one
      (keeping aspect ratio) before stacking.
    - The gap is filled with white by default; change --gap-color if needed.
"""

import argparse
import os

from PIL import Image


def parse_color(text):
    """Parse a color string like 'white', '#RRGGBB' or 'R,G,B' to an RGB tuple."""
    named = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (200, 200, 200),
        "grey": (200, 200, 200),
    }
    if text in named:
        return named[text]
    if text.startswith("#") and len(text) == 7:
        return tuple(int(text[i : i + 2], 16) for i in (1, 3, 5))
    if "," in text:
        parts = [int(x) for x in text.split(",")]
        if len(parts) == 3:
            return tuple(parts)
    raise ValueError(f"Unrecognized color: {text}")


def resize_to_width(img, target_w):
    if img.width == target_w:
        return img
    new_h = round(img.height * target_w / img.width)
    return img.resize((target_w, new_h), Image.LANCZOS)


def concat_pair(top_path, bottom_path, out_path, gap, gap_color):
    top = Image.open(top_path).convert("RGB")
    bot = Image.open(bottom_path).convert("RGB")

    # Match widths (in case renders differ slightly).
    target_w = max(top.width, bot.width)
    top = resize_to_width(top, target_w)
    bot = resize_to_width(bot, target_w)

    total_h = top.height + gap + bot.height
    canvas = Image.new("RGB", (target_w, total_h), gap_color)
    canvas.paste(top, (0, 0))
    canvas.paste(bot, (0, top.height + gap))
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Concatenate paired ranklist images (top/bottom).")
    parser.add_argument(
        "--top",
        default="/xxx/extrawork/code/svc_net/ranklist/market/baseline",
        help="Directory whose images are placed on TOP (baseline).",
    )
    parser.add_argument(
        "--bottom",
        default="/xxx/extrawork/code/svc_net/ranklist/market/market_01_oval_single",
        help="Directory whose images are placed on BOTTOM (ours).",
    )
    parser.add_argument(
        "--output",
        default="/xxx/extrawork/code/svc_net/ranklist/market/compare_baseline_vs_oval",
        help="Directory to save concatenated images.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=12,
        help="Vertical gap in pixels between top and bottom images.",
    )
    parser.add_argument(
        "--gap-color",
        type=str,
        default="white",
        help="Gap fill color: named ('white','black','gray'), '#RRGGBB', or 'R,G,B'.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".png",
        help="Image file extension to process (default: .png).",
    )
    args = parser.parse_args()

    gap_color = parse_color(args.gap_color)
    os.makedirs(args.output, exist_ok=True)

    top_files = {f for f in os.listdir(args.top) if f.lower().endswith(args.ext)}
    bot_files = {f for f in os.listdir(args.bottom) if f.lower().endswith(args.ext)}
    common = sorted(top_files & bot_files, key=lambda x: (len(x), x))

    only_top = top_files - bot_files
    only_bot = bot_files - top_files
    if only_top:
        print(f"[warn] {len(only_top)} files only in top dir, skipped: {sorted(only_top)[:5]}...")
    if only_bot:
        print(f"[warn] {len(only_bot)} files only in bottom dir, skipped: {sorted(only_bot)[:5]}...")

    if not common:
        print("[error] No shared filenames between the two directories.")
        return

    for name in common:
        top_path = os.path.join(args.top, name)
        bot_path = os.path.join(args.bottom, name)
        out_path = os.path.join(args.output, name)
        concat_pair(top_path, bot_path, out_path, args.gap, gap_color)
        print(f"[ok] {name}")

    print(f"\nDone. {len(common)} images saved to: {args.output}")


if __name__ == "__main__":
    main()
