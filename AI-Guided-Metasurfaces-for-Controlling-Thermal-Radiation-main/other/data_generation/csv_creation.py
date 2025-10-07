# -*- coding: utf-8 -*-
"""
This script processes COMSOL data files ('model*.txt') from a specified input folder,
extracts absorbance data, and compiles it into a single CSV file.

It's designed to be run from a standard Python environment on any operating system.
"""

import re
import csv
import argparse
from pathlib import Path

# --- Configuration ---

# Mapping from model number (as a string) to the shape name.
MODEL_TO_SHAPE_MAP = {
    '1': 'circle',
    '2': 'square',
    '3': 'blob_L_row',
    '4': 'rect_rot',
    '5': 'l_shape',
    '6': 'c_curved',
    '7': 'wedge',
    '8': 'cross',
    '9': 'polygon',
    '10': 'k_array_strict',
    '11': 'k_array_strict',
    '12': 'k_array_strict',
    '13': 'k_array_strict',
    '14': 'k_array_strict',
    '15': 'k_array_strict',
}

def parse_data_file(file_path: Path):
    """
    Reads a single COMSOL txt file and extracts absorbance values.

    Args:
        file_path (Path): The path object pointing to the input text file.

    Returns:
        A tuple containing (list_of_radians, list_of_absorbances).
        Returns (None, None) if the file cannot be processed.
    """
    radian_values = []
    absorbance_values = []
    try:
        with file_path.open('r') as f:
            for line in f:
                # Ignore comment/header lines starting with '%'
                if line.strip().startswith('%'):
                    continue

                # Split the line into columns and clean up whitespace
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        rad = float(parts[0])
                        absorbance = float(parts[1])
                        radian_values.append(rad)
                        absorbance_values.append(absorbance)
                    except ValueError:
                        print(f"Warning: Could not parse line in {file_path.name}: {line.strip()}")
                        continue
    except Exception as e:
        print(f"An error occurred while reading {file_path.name}: {e}")
        return None, None

    return radian_values, absorbance_values

def main():
    """
    Main function to find all data files, process them, and write the CSV.
    """
    # --- Set up Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Compile COMSOL absorbance data into a single CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('./Calculated'),
        help="The folder where your 'model...txt' files are located.\n(Default: ./Calculated)"
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        # Default path is set to go up two directories from the script's location.
        default=Path('../../metasurface_absorbance_compiled_final.csv'),
        help="The name of the CSV file that will be created.\n(Default: ../../metasurface_absorbance_compiled_final.csv)"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output_file

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found or is not a directory.")
        return

    # Regex to extract model and frame numbers from filenames
    # e.g., "model1_frame0120.txt" -> model='1', frame='0120'
    filename_pattern = re.compile(r"model(\d+)_frame(\d+)\.txt")

    all_data_rows = []
    header_generated = False
    csv_header = []

    # Get a sorted list of .txt files to process them in a consistent order
    files_to_process = sorted(input_dir.glob('model*.txt'))

    if not files_to_process:
        print(f"Error: No 'model*.txt' files found in directory '{input_dir}'.")
        return

    print(f"Found {len(files_to_process)} files. Starting processing...")

    for file_path in files_to_process:
        match = filename_pattern.match(file_path.name)
        if not match:
            print(f"Skipping file with unexpected name: {file_path.name}")
            continue

        model_num_str = match.group(1)
        frame_num_str = match.group(2)

        # Get the shape name from our map, with a fallback for unknown models
        shape_name = MODEL_TO_SHAPE_MAP.get(model_num_str, f"unknown_model_{model_num_str}")
        image_name = f"{shape_name}_{frame_num_str}.png"

        radians, absorbances = parse_data_file(file_path)

        if absorbances is None:
            continue # Skip file if parsing failed

        # Generate the header row from the first successfully processed file
        if not header_generated and radians:
            csv_header = ['Image Name'] + [f"Absorbance_{rad:.2f}" for rad in radians]
            header_generated = True

        # Create the data row for the CSV
        data_row = [image_name] + absorbances
        all_data_rows.append(data_row)

    if not all_data_rows:
        print("No data was processed successfully. Exiting.")
        return

    # Sort the collected data by the 'Image Name' column to ensure order
    all_data_rows.sort(key=lambda row: row[0])

    # Write all the collected data to the CSV file
    try:
        # Resolve the path to get a clean, absolute path for the output message
        absolute_output_path = output_path.resolve()
        with output_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(all_data_rows)
        print(f"\n✅ Successfully created CSV file: {absolute_output_path}")
        print(f"   Total rows written: {len(all_data_rows)}")
    except IOError as e:
        print(f"Error writing to CSV file '{output_path}': {e}")


# This makes the script runnable when called directly from the command line
if __name__ == "__main__":
    main()