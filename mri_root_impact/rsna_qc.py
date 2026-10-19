"""Render external source overlays and representative actual classifier views."""

import math

from .rsna_assets import file_sha256
from .rsna_geometry import LEVELS


def render_qc(frames, spatial, examples, output):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    used_ids = {rid for entry in spatial["inputs"] for ids in entry["roi_ids"].values() for rid in ids}
    colors = [(255, 204, 64), (66, 225, 215), (246, 144, 199), (146, 225, 101), (127, 177, 255)]
    font = ImageFont.load_default(size=11)
    header = 32
    folder = output / "overlays"
    folder.mkdir()
    overlays, thumbnails = [], []
    for index, frame in enumerate(frames):
        width, height = frame["columns"], frame["rows"]
        image = Image.new("RGB", (width, height + header), "#121821")
        image.paste(Image.fromarray(frame["pixels"]).convert("RGB"), (0, header))
        draw = ImageDraw.Draw(image)
        draw.text((3, 2), "MODEL ROI / UNREVIEWED", font=font, fill="white")
        draw.text((3, 16), "{} | inst {}".format(frame["sequence"], frame["instance_number"]), font=font, fill="white")
        for roi in spatial["rois"]:
            if roi["roi_id"] not in used_ids or roi["image_ref"] != frame["image_ref"]:
                continue
            left, top, right, bottom = roi["source_crop_box"]
            color = colors[LEVELS.index(roi["level"])]
            draw.rectangle((left, top + header, right - 1, bottom + header - 1), outline=color, width=1)
            draw.text((left + 1, max(header, top + header - 12)), roi["level"] + " " + roi["side"][0].upper(), font=font, fill=color)
        for point in spatial["keypoints"]:
            if point["image_ref"] == frame["image_ref"]:
                x, y = point["column"], point["row"] + header
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red")
                label = point.get("native_level", point["label"])
                draw.text((x + 3, y), label, font=font, fill="red")
        relative = "overlays/frame-{:03d}.png".format(index)
        image.save(output / relative)
        overlays.append({"file": relative, "image_ref": frame["image_ref"], "header_pixels": header,
                         "width": width, "height": height + header, "sha256": file_sha256(output / relative)})
        thumbnails.append(image.resize((256, 288)))
    columns = 5
    montage = Image.new("RGB", (columns * 256, max(1, math.ceil(len(thumbnails) / columns)) * 288), "#121821")
    for i, image in enumerate(thumbnails):
        montage.paste(image, ((i % columns) * 256, (i // columns) * 288))
    montage.save(output / "overlay.png")
    cell, label_height = 160, 28
    roi_montage = Image.new("RGB", (cell * 6, (cell + label_height) * 5), "#121821")
    draw = ImageDraw.Draw(roi_montage)
    for row, level in enumerate(LEVELS):
        for column, (sequence, side) in enumerate((s, side) for s in ("sagittal_t1", "sagittal_t2", "axial_t2") for side in ("left", "right")):
            x, y = column * cell, row * (cell + label_height)
            draw.text((x + 2, y + 2), level + " " + side, font=font, fill="white")
            draw.text((x + 2, y + 14), sequence + " | unreviewed", font=font, fill="white")
            array = examples.get((level, side, sequence))
            if array is not None:
                display = np.rint(np.clip((array + 1) * 127.5, 0, 255)).astype(np.uint8)
                roi_montage.paste(Image.fromarray(display).convert("RGB").resize((cell, cell)), (x, y + label_height))
            else:
                draw.text((x + 8, y + 50), "NOT RUN", font=font, fill="#FF8C70")
    roi_montage.save(output / "roi_montage.png")
    return overlays
