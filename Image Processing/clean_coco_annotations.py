import json
import os

def calculate_polygon_area(segmentation):
    # Handles polygon segmentations only
    if isinstance(segmentation, list):
        total_area = 0
        for poly in segmentation:
            if len(poly) < 6:  # Must have at least 3 points
                continue
            x = poly[::2]
            y = poly[1::2]
            area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(x) - 1)))
            total_area += area
        return total_area
    return 0
def clean_annotations(input_path, output_path, min_area_threshold=10):
    with open(input_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    cleaned_annotations = []

    # Group annotations by (image_id, category_id)
    grouped = {}
    for ann in annotations:
        key = (ann['image_id'], ann['category_id'])
        grouped.setdefault(key, []).append(ann)

    for (image_id, category_id), ann_list in grouped.items():
        if len(ann_list) == 1:
            cleaned_annotations.append(ann_list[0])
        else:
            # Keep the largest segmentation only (by area)
            anns_with_area = [
                (ann, calculate_polygon_area(ann['segmentation']))
                for ann in ann_list
            ]
            # Remove segmentations with area below threshold
            anns_with_area = [pair for pair in anns_with_area if pair[1] >= min_area_threshold]

            if anns_with_area:
                largest_ann = max(anns_with_area, key=lambda x: x[1])[0]
                cleaned_annotations.append(largest_ann)

    print(f"Original annotations: {len(annotations)}")
    print(f"Cleaned annotations:  {len(cleaned_annotations)}")

    data['annotations'] = cleaned_annotations

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved cleaned file to: {output_path}")

# --- Run it ---
if __name__ == "__main__":
    input_json = r'C:\Users\SR207348\Downloads\nnunet_output.json'  # Must be in same folder or adjust path
    output_json = "cleaned_nnunet_output.json"
    clean_annotations(input_json, output_json)
