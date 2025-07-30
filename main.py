import os
import sys
import pandas as pd
from pseudoaxisalignment import process_all_femoral_heads
from analyzer import FemoralHeadAnalyzer


# File Paths
coco_json_path = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\FragmentationStage\output.json'
image_folder = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\FragmentationStage'
img_output_folder = r'C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\Fragmentation Stage Measurements'
# Get the root directory of your project (where the script is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Set the output directory to "Data Analysis" subfolder
excel_output = os.path.join(PROJECT_ROOT, "Data Analysis")

# Ensure output directory exists
os.makedirs(excel_output, exist_ok=True)
os.makedirs(img_output_folder, exist_ok=True)

# Align Hips
aligned_results = process_all_femoral_heads(
    coco_json_path,
    image_folder,
    output_folder=img_output_folder,
    visualize=False,
    max_pairs=10
)

# Initialize results collector
all_results = []

# Process each result
for i, result in enumerate(aligned_results):
    patient_id = result['patient_id']
    timepoint = result['timepoint']
    print(f"Processing {patient_id}-{timepoint}...")

    analyzer = FemoralHeadAnalyzer(result, image_folder, img_output_folder)
    vis_path, results_dict = analyzer.process()
    all_results.append(results_dict)

# Export to Excel after all processing
if all_results:
    df = pd.DataFrame(all_results)
    excel_path = os.path.join(excel_output, "measurements_summary.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\nSaved measurements summary to {excel_path}")
else:
    print("No results to export")