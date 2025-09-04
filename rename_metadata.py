#!/usr/bin/env python3
"""
Script to rename metadata files to the standard 'metadata.csv' format
for LesionGen datasets
"""

import os
import glob
from pathlib import Path

def find_and_rename_metadata(directory, target_name="metadata.csv"):
    """Find and rename metadata files in a directory"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory {directory} does not exist")
        return False
    
    # Common metadata file patterns
    patterns = [
        "*.csv",
        "*metadata*.csv", 
        "*meta*.csv",
        "HAM10000_metadata*.csv",
        "D7P_metadata*.csv",
        "*_with_labels.csv"
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(str(directory / pattern)))
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))
    
    if not found_files:
        print(f"⚠️  No metadata files found in {directory}")
        return False
    
    if len(found_files) == 1:
        # Only one file found, rename it
        source_file = Path(found_files[0])
        target_file = directory / target_name
        
        if source_file.name == target_name:
            print(f"✅ {source_file} is already named correctly")
            return True
        
        try:
            source_file.rename(target_file)
            print(f"✅ Renamed {source_file.name} → {target_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to rename {source_file.name}: {e}")
            return False
    else:
        # Multiple files found, show options
        print(f"📁 Found multiple metadata files in {directory}:")
        for i, file_path in enumerate(found_files, 1):
            print(f"  {i}. {Path(file_path).name}")
        
        # Try to find the most likely candidate
        likely_candidates = [f for f in found_files if "metadata" in Path(f).name.lower()]
        if likely_candidates:
            source_file = Path(likely_candidates[0])
            target_file = directory / target_name
            
            if source_file.name != target_name:
                try:
                    source_file.rename(target_file)
                    print(f"✅ Auto-renamed {source_file.name} → {target_name}")
                    return True
                except Exception as e:
                    print(f"❌ Failed to rename {source_file.name}: {e}")
                    return False
            else:
                print(f"✅ {source_file.name} is already named correctly")
                return True
        
        print(f"⚠️  Please manually rename one of the files to '{target_name}'")
        return False

def main():
    print("🔄 LesionGen Metadata File Renamer")
    print("=" * 40)
    
    # Define dataset directories
    datasets = {
        "HAM10000": "data/ham10000",
        "D7P": "data/d7p"
    }
    
    success_count = 0
    
    for dataset_name, directory in datasets.items():
        print(f"\n📂 Processing {dataset_name} dataset...")
        if find_and_rename_metadata(directory):
            success_count += 1
    
    print(f"\n📊 Summary: {success_count}/{len(datasets)} datasets processed successfully")
    
    if success_count == len(datasets):
        print("\n✅ All metadata files renamed successfully!")
        print("🚀 You can now run the training scripts:")
        print("   ./train_lora.sh")
        print("   ./train_SD.sh")
    else:
        print("\n⚠️  Some metadata files could not be renamed automatically.")
        print("Please check the directories and rename manually if needed.")

if __name__ == "__main__":
    main()
