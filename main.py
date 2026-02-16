import os
import sys
import pandas as pd
import time
from pseudoaxisalignment import process_all_femoral_heads
from analyzer import FemoralHeadAnalyzer

# File Paths
coco_json_path = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\FragmentationStage\output.json'
image_folder = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\FragmentationStage'
img_output_folder = r'Y:\Clinical Research\KIM\STUDENTS\Kevin Li\HIPMETRICS\TSRH_viz'
# Get the root directory of your project (where the script is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Set the output directory to "Data Analysis" subfolder
excel_output = os.path.join(PROJECT_ROOT, "Data Analysis")

# Ensure output directory exists
os.makedirs(excel_output, exist_ok=True)
os.makedirs(img_output_folder, exist_ok=True)

# Align Hips - REMOVED max_pairs parameter to process all
aligned_results = process_all_femoral_heads(
    coco_json_path,
    image_folder,
    output_folder=img_output_folder,
    visualize=False,
    #max_pairs = 10

)

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