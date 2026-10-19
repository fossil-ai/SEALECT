import json
from collections import defaultdict

COCO_JSON_PATH = "/path/to/datasets/data/merged/annotations_fixed.json"  # <-- change this


def main():
    with open(COCO_JSON_PATH, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Map image_id -> list of sources (from annotations)
    image_sources = defaultdict(set)
    source_box_counts = defaultdict(int)

    # Count boxes per source
    for ann in annotations:
        source = ann.get("source", "unknown")
        image_id = ann["image_id"]

        source_box_counts[source] += 1
        image_sources[image_id].add(source)

    # Count images per source
    source_image_counts = defaultdict(int)

    for image_id, sources in image_sources.items():
        for s in sources:
            source_image_counts[s] += 1

    # Handle images with no annotations
    annotated_image_ids = set(image_sources.keys())
    all_image_ids = set(img["id"] for img in images)

    empty_images = all_image_ids - annotated_image_ids

    if empty_images:
        source_image_counts["no_annotations"] = len(empty_images)

    # Totals
    total_images = len(images)
    total_boxes = len(annotations)

    # Print results
    print("\n=== DATASET BREAKDOWN ===\n")

    sources = set(source_box_counts.keys()) | set(source_image_counts.keys())

    for s in sorted(sources):
        print(f"{s:20s} | images: {source_image_counts[s]:6d} | boxes: {source_box_counts[s]:6d}")

    print("\n=== TOTALS ===")
    print(f"Total images: {total_images}")
    print(f"Total boxes:  {total_boxes}")


if __name__ == "__main__":
    main()