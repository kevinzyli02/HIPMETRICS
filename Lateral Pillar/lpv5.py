import os
from pseudoaxisalignment import process_all_femoral_heads
from PerthesMeasurementIntermediate import FemoralHeadAnalyzer

# File Paths for COCO_JSON(Annotations), image folder, and output destination
coco_json_path = r'C:\Users\SR207348\Downloads\labels_ipsg102_2025-06-30-08-40-52.json'
image_folder = r'C:\Users\SR207348\Downloads\ipsg102\ipsg102'
output_folder = r"C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\ipsg102_DI"
os.makedirs(output_folder, exist_ok=True)

# Align Hips: Set visualize=True if you want alignment results, and change max_pairs if you want to limit how many are analyzed
aligned_results = process_all_femoral_heads(
        coco_json_path,
        image_folder,
        output_folder=output_folder,
        visualize=False,
        max_pairs=10  # Only process 10 hips
    )

# Initialize and calculate measurements
for result in aligned_results:
    analyzer = FemoralHeadAnalyzer(result, output_folder)
    analyzer.process()





