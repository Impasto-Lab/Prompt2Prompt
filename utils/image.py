"""Turn decoded pixels into PIL images and lay them out as a captioned grid."""
from PIL import Image, ImageDraw, ImageFont


def to_pil(pixels):
    """Batch of images in roughly [-1, 1], shape (B, 3, H, W) -> list of PIL images."""
    pixels = ((pixels.float() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu()
    return [Image.fromarray((p.numpy() * 255).round().astype("uint8")) for p in pixels]


def load_font(size):
    # Serif captions, as in the paper figure; fall back to PIL's built-in font.
    for name in ("times.ttf", "Times New Roman.ttf", "DejaVuSerif.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def make_grid(images, captions, cols, gap=8, caption_height=40):
    """Lay out images row by row with a caption centred under each one."""
    w, h = images[0].size
    rows = -(-len(images) // cols)
    cell_h = h + caption_height
    grid = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap), "white")
    draw = ImageDraw.Draw(grid)
    font = load_font(caption_height * 11 // 20)
    for i, (image, caption) in enumerate(zip(images, captions)):
        x = gap + (i % cols) * (w + gap)
        y = gap + (i // cols) * (cell_h + gap)
        grid.paste(image, (x, y))
        draw.text((x + w / 2, y + h + caption_height / 2), caption, fill="black", font=font, anchor="mm")
    return grid
