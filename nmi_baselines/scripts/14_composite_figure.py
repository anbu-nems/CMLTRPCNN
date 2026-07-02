"""
14_composite_figure.py — composite the 5 standalone NMI main figures into one
combined image using PIL. Avoids matplotlib layout battles by pasting
pre-rendered PNGs at consistent widths.

Layout (3 rows, total width = max-width of any constituent figure):
    Row 1: fig_NMI_1_fLST_hierarchy        (double width 2-panel)
    Row 2: fig_NMI_2_baseline_comparison   (double width 2-panel)
    Row 3: fig_NMI_3 | fig_NMI_4 | fig_NMI_5  (three single-width panels side-by-side)

Each row is normalised to the same total width by scaling sub-images
proportionally. White background padding between rows for breathing room.

Output: fig_NMI_combined_composite.{png,pdf}
"""
import os
from PIL import Image

FIG_DIR = "../figures"

# Input PNGs
F1 = os.path.join(FIG_DIR, "fig_NMI_1_fLST_hierarchy.png")
F2 = os.path.join(FIG_DIR, "fig_NMI_2_baseline_comparison.png")
F3 = os.path.join(FIG_DIR, "fig_NMI_3_lofo_sign_guarantee.png")
F4 = os.path.join(FIG_DIR, "fig_NMI_4_variants_accuracy.png")
F5 = os.path.join(FIG_DIR, "fig_NMI_5_variants_uq.png")

# Load all
img1, img2 = Image.open(F1), Image.open(F2)
img3, img4, img5 = Image.open(F3), Image.open(F4), Image.open(F5)
print(f"[loaded] sizes: F1={img1.size}, F2={img2.size}, F3={img3.size}, F4={img4.size}, F5={img5.size}")


# ── Decide a uniform "TARGET WIDTH" for the composite (use F1's width as reference) ──
TARGET_WIDTH = max(img1.width, img2.width)
print(f"[target] composite width = {TARGET_WIDTH}px")


def resize_to_width(img, target_w):
    """Resize PIL image to target width, maintaining aspect ratio."""
    w, h = img.size
    new_h = int(h * target_w / w)
    return img.resize((target_w, new_h), Image.LANCZOS)


# Resize each PNG so all rows can sit at the same width
img1_r = resize_to_width(img1, TARGET_WIDTH)
img2_r = resize_to_width(img2, TARGET_WIDTH)

# Row 3: 3 figures side-by-side, equal widths summing to TARGET_WIDTH (minus gaps)
GAP_BETWEEN_ROW3_PANELS = 10
sub_w = (TARGET_WIDTH - 2 * GAP_BETWEEN_ROW3_PANELS) // 3
img3_r = resize_to_width(img3, sub_w)
img4_r = resize_to_width(img4, sub_w)
img5_r = resize_to_width(img5, sub_w)

# Row 3 height = max of the three (they should all be similar after equal-width scaling)
row3_h = max(img3_r.height, img4_r.height, img5_r.height)

# Compose
ROW_GAP = 25
total_h = img1_r.height + ROW_GAP + img2_r.height + ROW_GAP + row3_h
composite = Image.new("RGB", (TARGET_WIDTH, total_h), (255, 255, 255))

# Row 1
y = 0
composite.paste(img1_r, (0, y))
y += img1_r.height + ROW_GAP

# Row 2
composite.paste(img2_r, (0, y))
y += img2_r.height + ROW_GAP

# Row 3 — center-align each panel within its slot vertically
x = 0
for sub in (img3_r, img4_r, img5_r):
    y_offset = y + (row3_h - sub.height) // 2
    composite.paste(sub, (x, y_offset))
    x += sub_w + GAP_BETWEEN_ROW3_PANELS

# Save
png_out = os.path.join(FIG_DIR, "fig_NMI_combined_composite.png")
pdf_out = os.path.join(FIG_DIR, "fig_NMI_combined_composite.pdf")
composite.save(png_out, dpi=(300, 300))
# Convert to PDF via PIL
composite.save(pdf_out, "PDF", resolution=300)

print(f"\n[done] composite saved:")
print(f"  {png_out}  ({composite.size[0]}×{composite.size[1]}px)")
print(f"  {pdf_out}")
