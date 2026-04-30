import os
import re
import shutil
from pathlib import Path
from collections import defaultdict


def extract_image_info(filename):
    """
    Extract patient number, laterality, and month from filename.
    Format: Patient_31_AP_24_month_L.bmp
    """
    # Remove .bmp extension
    name = filename.replace('.bmp', '').replace('.BMP', '')

    # Pattern: Patient_<number>_<view>_<month>_month_<laterality>
    pattern = r'Patient_(\d+)_([A-Za-z]+)_(\d+)_month_([LR])'
    match = re.match(pattern, name)

    if match:
        patient_num = int(match.group(1))
        view = match.group(2)
        month = int(match.group(3))
        laterality = match.group(4)
        return {
            'patient': patient_num,
            'view': view,
            'month': month,
            'laterality': laterality,
            'filename': filename
        }
    return None


def select_best_images(images_by_patient):
    """
    For each patient, select 1 Left and 1 Right image.
    Prefer 24-month, otherwise closest to 24.
    """
    selected = []

    for patient, images in sorted(images_by_patient.items()):
        # Group by laterality
        by_laterality = defaultdict(list)
        for img in images:
            by_laterality[img['laterality']].append(img)

        # For each side, select the best month
        for laterality in ['L', 'R']:
            if laterality in by_laterality:
                candidates = by_laterality[laterality]
                # Sort by distance from 24, then by month value
                best = min(candidates, key=lambda x: (abs(x['month'] - 24), x['month']))
                selected.append(best)

    return selected


def filter_images_and_masks(source_dir, output_dir):
    """
    Main filtering function
    """
    # Statistics tracking
    stats = {
        'frog_skipped': 0,
        'non_ap_skipped': 0,
        'parse_errors': 0,
        'masks_by_type': defaultdict(int),
        'total_masks_copied': 0,
    }
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Scan BMP files in test folder
    images_by_patient = defaultdict(list)
    all_image_files = []

    print(f"Scanning images in: {source_dir}")
    for filename in os.listdir(source_dir):
        filepath = os.path.join(source_dir, filename)

        # Skip directories
        if os.path.isdir(filepath):
            continue

        # Only process BMP files
        if not filename.lower().endswith('.bmp'):
            continue

        # Skip frog images and non-AP
        if 'frog' in filename.lower():
            stats['frog_skipped'] += 1
            continue

        info = extract_image_info(filename)
        if info and info['view'].upper() == 'AP':
            images_by_patient[info['patient']].append(info)
            all_image_files.append(filename)
        else:
            if info is None:
                stats['parse_errors'] += 1
            else:
                stats['non_ap_skipped'] += 1

    # Select best images
    selected_images = select_best_images(images_by_patient)
    selected_filenames = {img['filename'] for img in selected_images}

    print(f"\n{'=' * 60}")
    print(f"BEFORE FILTERING STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total unique patients: {len(images_by_patient)}")
    print(f"Total AP images (before filtering): {len(all_image_files)}")
    print(f"Images skipped - Frog: {stats['frog_skipped']}")
    print(f"Images skipped - Non-AP: {stats['non_ap_skipped']}")
    print(f"Images with parse errors: {stats['parse_errors']}")

    print(f"\n{'=' * 60}")
    print(f"FILTERING CRITERIA")
    print(f"{'=' * 60}")
    print(f"Selecting 2 images per patient (1L, 1R)")
    print(f"Preference: 24-month, or closest to 24-month")

    print(f"\n{'=' * 60}")
    print(f"AFTER FILTERING STATISTICS")
    print(f"{'=' * 60}")
    unique_patients_selected = len(set(img['patient'] for img in selected_images))
    print(f"Total unique patients selected: {unique_patients_selected}")
    print(f"Total AP images after filtering: {len(selected_filenames)}")
    print(f"Images removed: {len(all_image_files) - len(selected_filenames)}")

    print(f"\nSelected images breakdown:")
    laterality_counts = defaultdict(int)
    for img in selected_images:
        laterality_counts[img['laterality']] += 1
    for laterality in ['L', 'R']:
        if laterality in laterality_counts:
            print(f"  - {laterality} (Right): {laterality_counts[laterality]}")

    print(f"\nSelected images:")
    for img in sorted(selected_filenames):
        print(f"  ✓ {img}")

    # Copy selected images to output
    print(f"\n{'=' * 60}")
    print(f"COPYING FILES")
    print(f"{'=' * 60}")
    print(f"Copying {len(selected_filenames)} images to output...")
    images_copied = 0
    for filename in selected_filenames:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            images_copied += 1

    print(f"✓ Copied {images_copied} images")

    # Handle masks
    masks_dir = os.path.join(source_dir, 'masks')

    if os.path.exists(masks_dir):
        print(f"\nScanning mask subfolders in: {masks_dir}")

        # Get list of mask type directories (acetabulum, gt, head, lt, etc)
        mask_types = []
        for item in os.listdir(masks_dir):
            item_path = os.path.join(masks_dir, item)
            if os.path.isdir(item_path):
                mask_types.append(item)

        print(f"Found mask types: {sorted(mask_types)}")

        # For each mask type, copy matching masks
        for mask_type in sorted(mask_types):
            mask_type_dir = os.path.join(masks_dir, mask_type)
            output_mask_dir = os.path.join(output_dir, 'masks', mask_type)
            os.makedirs(output_mask_dir, exist_ok=True)

            masks_in_type = 0
            for mask_filename in os.listdir(mask_type_dir):
                if mask_filename in selected_filenames:
                    src = os.path.join(mask_type_dir, mask_filename)
                    dst = os.path.join(output_mask_dir, mask_filename)

                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        masks_in_type += 1
                        stats['total_masks_copied'] += 1

            stats['masks_by_type'][mask_type] = masks_in_type
            print(f"  {mask_type}: {masks_in_type} masks copied")
    else:
        print(f"\nWARNING: Masks folder not found at: {masks_dir}")

    print(f"\n{'=' * 60}")
    print(f"MASK COPYING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total masks copied: {stats['total_masks_copied']}")
    print(f"Masks per type:")
    for mask_type in sorted(stats['masks_by_type'].keys()):
        count = stats['masks_by_type'][mask_type]
        print(f"  - {mask_type}: {count}")
    
    expected_masks_per_type = len(selected_filenames)
    print(f"\nExpected masks per type: {expected_masks_per_type}")
    print(f"Verification:")
    all_correct = True
    for mask_type in sorted(stats['masks_by_type'].keys()):
        count = stats['masks_by_type'][mask_type]
        if count == expected_masks_per_type:
            print(f"  ✓ {mask_type}: Correct ({count} masks)")
        else:
            print(f"  ✗ {mask_type}: MISMATCH (expected {expected_masks_per_type}, got {count})")
            all_correct = False

    print(f"\n{'=' * 60}")
    print(f"✓ FILTERING COMPLETE!")
    print(f"{'=' * 60}")
    if all_correct:
        print(f"✓ All masks correctly copied!")
    else:
        print(f"⚠ Some mask counts don't match - check output!")
    print(f"Output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Update these paths to your actual locations
    source_dir = r"\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\Intermediate_stage\radiograph_dataset_18month-30month\train"
    output_dir = r"\\wnresearch\Drobo\Vishal_Graham\ML Review\radiographs\Intermediate_stage\radiograph_dataset_18month-30month\filtered_train"

    try:
        filter_images_and_masks(source_dir, output_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
