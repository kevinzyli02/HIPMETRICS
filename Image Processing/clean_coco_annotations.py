import json
import os
import glob


def calculate_polygon_area(segmentation):
    # Handles polygon segmentations only
    if isinstance(segmentation, list):
        total_area = 0
        for poly in segmentation:
            if len(poly) < 6:  # Must have at least 3 points
                continue
            x = poly[::2]
            y = poly[1::2]
            # Fixed shoelace formula with proper wraparound
            area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(x)] - x[(i + 1) % len(x)] * y[i]
                                 for i in range(len(x))))
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

    print(f"  Original annotations: {len(annotations)}")
    print(f"  Cleaned annotations:  {len(cleaned_annotations)}")

    data['annotations'] = cleaned_annotations

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def process_folder(folder_path, min_area_threshold=10):
    """Process all JSON files in a folder"""
    # Find all JSON files in the folder
    json_pattern = os.path.join(folder_path, "*.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"No JSON files found in: {folder_path}")
        return

    print(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        try:
            # Generate output filename with _cleaned suffix
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            output_name = f"{base_name}_cleaned.json"
            output_path = os.path.join(folder_path, output_name)

            print(f"\nProcessing: {os.path.basename(json_file)}")
            clean_annotations(json_file, output_path, min_area_threshold)
            print(f"  Saved to: {output_name}")

        except Exception as e:
            print(f"  Error processing {os.path.basename(json_file)}: {str(e)}")


def process_single_file(input_path, min_area_threshold=10):
    """Process a single JSON file"""
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    # Generate output filename with _cleaned suffix
    directory = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_name = f"{base_name}_cleaned.json"
    output_path = os.path.join(directory, output_name)

    print(f"Processing: {os.path.basename(input_path)}")
    clean_annotations(input_path, output_path, min_area_threshold)
    print(f"Saved to: {output_path}")


# --- Run it ---
if __name__ == "__main__":
    # Configuration
    min_area_threshold = 1

    # Option 1: Process entire folder
    folder_path = r'Y:\Clinical Research\KIM\STUDENTS\Kevin Li\Alexs JSON'  # Change this to your folder path
    process_folder(folder_path, min_area_threshold)

    # Option 2: Process single file (comment out the above and uncomment below)
    # input_json = r'C:\Users\SR207348\Downloads\nnunet_output.json'
    # process_single_file(input_json, min_area_threshold)