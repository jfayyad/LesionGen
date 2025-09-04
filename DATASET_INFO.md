# Dataset Information for LesionGen

## 📊 Available Datasets

### HAM10000 Dataset
- **Full Dataset**: Available from the official source
- **Source**: [Nature Scientific Data](https://www.nature.com/articles/sdata2018161)
- **Size**: 10,015 dermatoscopic images
- **Classes**: 7 different types of skin lesions
- **Download**: Contact the authors or visit the official dataset page

### D7P Dataset
- **Full Dataset**: Contact the original authors
- **Size**: Additional dermatological images
- **Classes**: Various skin lesion types
- **Download**: Contact the original paper authors

## 📁 Metadata Files

The Google Drive links provided in the repository contain **metadata files only**, not the full datasets with images:

- **HAM10000 Metadata**: [Download Link](https://drive.google.com/file/d/1K5oGP55B5d9lhhFjzGTJ4kgtIdbnSCmg/view?usp=sharing)
- **D7P Metadata**: [Download Link](https://drive.google.com/file/d/1_56PsBov6rI6_F9JfBf_2GKd8hQolA3Y/view?usp=sharing)

## 🚀 How to Get Started

### Download Everything Separately
1. **Download metadata files** using the provided Google Drive links
2. **Download full datasets** from their original sources
3. **Extract datasets** to `data/ham10000/` and `data/d7p/` directories
4. **Rename metadata files** to `metadata.csv` in their respective directories


## 📋 Required Directory Structure

```
data/
├── ham10000/
│   ├── images/           # Full HAM10000 images (from original source)
│   └── metadata.csv      # Metadata file (from Google Drive)
└── d7p/
    ├── images/           # Full D7P images (from original source)
    └── metadata.csv      # Metadata file (from Google Drive)
```
