from pathlib import Path
from PIL import Image
import json
import copy

# -------- CONFIG --------
IMAGE_DIR = Path("/path/to/sealect_root/data/images")
JSON_DIR = Path("/path/to/sealect_root/data/annotations1")

OUT_IMAGE_DIR = Path("/path/to/sealect_root/data/images_aug")
OUT_JSON_DIR = Path("/path/to/sealect_root/data/jsons_aug")

SUFFIX = "_hflip"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ------------------------


def wrap_angle_deg(angle):
    """Wrap angle to [-180, 180)."""
    return ((angle + 180) % 360) - 180


def flip_x(x, image_width):
    return image_width - x


def augment_json_horizontal_flip(data, image_width, image_height, new_image_name):
    aug = copy.deepcopy(data)

    aug["imageFileName"] = new_image_name

    for ann in aug.get("annotations", []):
        # Flip bounding box
        bbox = ann.get("bounding_box")
        if bbox:
            old_x = bbox["x"]
            old_w = bbox["width"]
            bbox["x"] = image_width - old_x - old_w

        # Flip stern/bow points if present and nonzero
        for key in ["stern_pixel", "bow_pixel"]:
            pt = ann.get(key)
            if pt and not (pt.get("x", 0.0) == 0.0 and pt.get("y", 0.0) == 0.0):
                pt["x"] = flip_x(pt["x"], image_width)

        # Flip segmentation if present
        # Supports either [{x,y}, ...] or [[x,y], ...]
        for seg_key in ["segmentation", "segmentation_ui"]:
            seg = ann.get(seg_key)
            if isinstance(seg, list):
                for p in seg:
                    if isinstance(p, dict) and "x" in p:
                        p["x"] = flip_x(p["x"], image_width)
                    elif isinstance(p, list) and len(p) >= 2:
                        p[0] = flip_x(p[0], image_width)

        # Flip pose
        pose = ann.get("pose")
        if pose:
            pose["yaw"] = wrap_angle_deg(-pose.get("yaw", 0.0))
            pose["pitch"] = pose.get("pitch", 0.0)
            pose["roll"] = wrap_angle_deg(-pose.get("roll", 0.0))

    return aug


def main():
    OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    flipped_count = 0
    skipped_json_count = 0

    for img_path in images:
        json_path = JSON_DIR / f"{img_path.stem}.json"

        if not json_path.exists():
            print(f"Skipping {img_path.name}: no matching JSON")
            skipped_json_count += 1
            continue

        new_img_name = f"{img_path.stem}{SUFFIX}{img_path.suffix}"
        new_json_name = f"{img_path.stem}{SUFFIX}.json"

        out_img_path = OUT_IMAGE_DIR / new_img_name
        out_json_path = OUT_JSON_DIR / new_json_name

        with Image.open(img_path) as img:
            width, height = img.size
            flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped.save(out_img_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        aug_data = augment_json_horizontal_flip(
            data=data,
            image_width=width,
            image_height=height,
            new_image_name=new_img_name,
        )

        with open(out_json_path, "w") as f:
            json.dump(aug_data, f, indent=4)

        flipped_count += 1
        print(f"Created {new_img_name} and {new_json_name}")

    print("\nDone.")
    print(f"Flipped image/json pairs: {flipped_count}")
    print(f"Skipped images without JSON: {skipped_json_count}")


if __name__ == "__main__":
    main()