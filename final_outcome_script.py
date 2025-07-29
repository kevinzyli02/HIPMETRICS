import os
import pandas as pd
import json
from collections import defaultdict

# Configuration - Fixed paths
COCO_JSON_PATH = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\FragmentationStage\output.json'
IMAGE_FOLDER = r'\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\FragmentationStage'
STULBERG_EXCEL_PATH = r"\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\Stulberg classification model - skeletal maturity\Classification Results\Stulberg classification.xlsx"
OUTPUT_EXCEL = r'C:\Users\SR207348\OneDrive - Scottish Rite for Children\Documents\Radiographic Annotations\patient_database.xlsx'


def extract_patient_data():
    """Extract patient information from COCO JSON"""
    try:
        with open(COCO_JSON_PATH, 'r') as f:
            coco_data = json.load(f)
        print(f"Successfully loaded COCO JSON from {COCO_JSON_PATH}")
    except FileNotFoundError:
        print(f"Error: COCO JSON file not found at {COCO_JSON_PATH}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {COCO_JSON_PATH}")
        return pd.DataFrame()

    # Create mapping from image_id to filename
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # Extract unique patients and count views
    patient_stats = defaultdict(lambda: {'ap_count': 0, 'frog_count': 0, 'timepoints': set()})

    # Track processed images to avoid duplicates
    processed_images = set()

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in processed_images:
            continue  # Skip duplicate annotations for same image

        processed_images.add(image_id)
        filename = image_id_to_name[image_id]
        patient_id, view = parse_filename(filename)

        if view == 'AP':
            patient_stats[patient_id]['ap_count'] += 1
        elif view == 'Frog':
            patient_stats[patient_id]['frog_count'] += 1

        # Extract timepoint from filename (third component)
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts) >= 3:
            patient_stats[patient_id]['timepoints'].add(parts[2])

    # Convert to structured data
    patient_data = []
    for pid, stats in patient_stats.items():
        patient_data.append({
            'patient_id': pid,
            'total_xrays': stats['ap_count'] + stats['frog_count'],
            'ap_count': stats['ap_count'],
            'frog_count': stats['frog_count'],
            'timepoints': len(stats['timepoints'])
        })

    return pd.DataFrame(patient_data)


def parse_filename(filename):
    """Extract patient ID and view from filename (handles two formats)"""
    base = os.path.splitext(filename)[0]
    parts = base.split('_')

    # Handle "Patient_X" format
    if parts[0].lower() == 'patient':
        patient_id = parts[1] if len(parts) > 1 else 'unknown'
        view_part = parts[2].lower() if len(parts) > 2 else ''
    # Handle "X_view" format
    else:
        patient_id = parts[0]
        view_part = parts[1].lower() if len(parts) > 1 else ''

    # Normalize view
    if 'ap' in view_part:
        view = 'AP'
    elif 'frog' in view_part:
        view = 'Frog'
    else:
        view = 'Unknown'

    return patient_id, view


def merge_with_stulberg(patient_df):
    """Merge patient data with Stulberg classifications using filename parsing"""
    try:
        # Read Stulberg Excel file
        stulberg_df = pd.read_excel(STULBERG_EXCEL_PATH)
        print("Columns in Stulberg file:", stulberg_df.columns.tolist())

        # Extract patient IDs from BMP filenames
        def extract_id_from_filename(filename):
            if not isinstance(filename, str):
                return None
            base = os.path.splitext(os.path.basename(filename))[0]
            parts = base.split('_')
            return parts[1] if parts[0].lower() == 'patient' else parts[0]

        stulberg_df['patient_id'] = stulberg_df['BMP file name'].apply(extract_id_from_filename)
        stulberg_df = stulberg_df.dropna(subset=['patient_id'])

    except Exception as e:
        print(f"Error processing Stulberg file: {e}")
        return patient_df

    # Find Stulberg classification column
    stulberg_col = next((col for col in stulberg_df.columns
                         if 'stulberg' in col.lower() or 'observer' in col.lower()), None)

    if not stulberg_col:
        print("Stulberg classification column not found. Using first observer column.")
        stulberg_col = stulberg_df.columns[1]  # Default to first data column

    # Merge data
    merged_df = pd.merge(
        patient_df,
        stulberg_df[['patient_id', stulberg_col]],
        on='patient_id',
        how='left'
    )
    merged_df.rename(columns={stulberg_col: 'stulberg_classification'}, inplace=True)
    return merged_df

def main():
    print("Starting patient data processing...")
    print(f"COCO JSON path: {COCO_JSON_PATH}")
    print(f"Stulberg Excel path: {STULBERG_EXCEL_PATH}")
    print(f"Output will be saved to: {OUTPUT_EXCEL}")

    # Extract patient data from COCO JSON
    patient_df = extract_patient_data()

    if patient_df.empty:
        print("No patient data extracted. Exiting.")
        return

    print(f"\nExtracted data for {len(patient_df)} patients")
    print("Sample of patient data:")
    print(patient_df.head())

    # Merge with Stulberg classifications
    final_db = merge_with_stulberg(patient_df)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)

    # Save final database
    final_db.to_excel(OUTPUT_EXCEL, index=False)

    print(f"\nPatient database created at: {OUTPUT_EXCEL}")
    print(f"Total patients processed: {len(final_db)}")
    print("\nFinal database columns:", final_db.columns.tolist())
    print("\nFirst 5 rows of final database:")
    print(final_db.head())

    # Summary statistics
    print("\nSummary:")
    print(f"- Total patients: {len(final_db)}")
    print(f"- Patients with Stulberg classification: {final_db['stulberg_classification'].count()}")
    print(f"- AP X-rays: {final_db['ap_count'].sum()}")
    print(f"- Frog X-rays: {final_db['frog_count'].sum()}")
    print(f"- Total X-rays: {final_db['total_xrays'].sum()}")
    print(f"- Average timepoints per patient: {final_db['timepoints'].mean():.1f}")


if __name__ == '__main__':
    main()