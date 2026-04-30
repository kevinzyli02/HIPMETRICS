import os
import sys
import pandas as pd
import time
from pseudoaxisalignment import process_all_femoral_heads, process_all_femoral_heads_bmp
from analyzer import FemoralHeadAnalyzer

# ── Data mode ─────────────────────────────────────────────────────────────────
# 'bmp'  – BMP-mask dataset (images + masks/head/ + masks/neck/ subfolders)
# 'coco' – COCO JSON-annotated dataset
DATA_MODE = 'bmp'

# ── BMP dataset paths ─────────────────────────────────────────────────────────
bmp_image_folder = r'C:\Users\SR207348\OneDrive - Scottish Rite for Children\Kim Research\HIPMETRICS-lateral pillar\DI images\filtered_test'

# ── COCO dataset paths ────────────────────────────────────────────────────────
coco_json_path = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\FragmentationStage\output.json'
coco_image_folder = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\FragmentationStage'

# ── Output paths ──────────────────────────────────────────────────────────────
img_output_folder = r'Y:\Clinical Research\KIM\STUDENTS\Kevin Li\HIPMETRICS\TSRH_viz'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
excel_output = os.path.join(PROJECT_ROOT, "Data Analysis")

os.makedirs(excel_output, exist_ok=True)
os.makedirs(img_output_folder, exist_ok=True)

# ── Align hips ────────────────────────────────────────────────────────────────
if DATA_MODE == 'bmp':
    aligned_results = process_all_femoral_heads_bmp(
        bmp_image_folder,
        output_folder=img_output_folder,
        visualize=False,
    )
    image_folder = bmp_image_folder
else:
    aligned_results = process_all_femoral_heads(
        coco_json_path,
        coco_image_folder,
        output_folder=img_output_folder,
        visualize=False,
        # max_pairs=10
    )
    image_folder = coco_image_folder

# Initialize results collector
all_results = []
processing_times = []  # Add this list to store processing times

# Process each result
for i, result in enumerate(aligned_results):
    patient_id = result['patient_id']
    timepoint = result['timepoint']
    print(f"Processing {patient_id}-{timepoint}...")

    start_time = time.time()  # Start timing

    analyzer = FemoralHeadAnalyzer(result, image_folder, img_output_folder)
    vis_path, results_dict = analyzer.process()
    all_results.append(results_dict)

    end_time = time.time()  # End timing
    processing_time = end_time - start_time
    processing_times.append(processing_time)
    print(f"Processed {patient_id}-{timepoint} in {processing_time:.2f} seconds")

# Export to Excel after all processing
if all_results:
    df = pd.DataFrame(all_results)
    excel_path = os.path.join(excel_output, "measurements_summary.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\nSaved measurements summary to {excel_path}")

    # Calculate and display timing statistics
    if processing_times:
        total_time = sum(processing_times)
        avg_time = total_time / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)

        print(f"\nTiming Statistics:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time:.2f} seconds")
        print(f"Fastest processing: {min_time:.2f} seconds")
        print(f"Slowest processing: {max_time:.2f} seconds")
        print(f"Number of images processed: {len(processing_times)}")
else:
    print("No results to export")