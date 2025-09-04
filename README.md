# LesionGen: AI-Powered Skin Lesion Generation

LesionGen is a comprehensive framework for generating synthetic skin lesion images using diffusion models. The project includes both full fine-tuning and LoRA (Low-Rank Adaptation) approaches for training Stable Diffusion models on dermatological datasets.

## 🎯 Features

- **Full Model Fine-tuning**: Complete Stable Diffusion model training on skin lesion datasets
- **LoRA Training**: Efficient fine-tuning using Low-Rank Adaptation
- **Classification Pipeline**: Train and evaluate skin lesion classifiers
- **Flexible Generation**: Generate single images or entire datasets
- **Multiple Datasets**: Support for HAM10000 and D7P datasets

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch` and `torchvision`
- `diffusers` (via submodule)
- `accelerate` for distributed training
- `transformers`
- `peft` for LoRA training

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LesionGen.git
cd LesionGen
```

### 2. Install Dependencies
```bash
# Install main requirements
pip install -r requirements.txt

# Install diffusers submodule
cd external/diffusers
pip install .
cd ../..
```

### 3. Download Datasets

#### HAM10000 Dataset
1. **Download the full HAM10000 dataset** from the official source:
   - Visit: [HAM10000 Dataset](https://www.nature.com/articles/sdata2018161)
   - Download the complete dataset with images
   - Extract to `data/ham10000/` directory

2. **Download the metadata file**:
   - **Metadata Link**: [HAM10000 Metadata](https://drive.google.com/file/d/1K5oGP55B5d9lhhFjzGTJ4kgtIdbnSCmg/view?usp=sharing)
   - Save as `metadata.csv` in the `data/ham10000/` folder

#### D7P Dataset
1. **Download the full D7P dataset** (contact the authors or check the original paper)
2. **Download the metadata file**:
   - **Metadata Link**: [D7P Metadata](https://drive.google.com/file/d/1_56PsBov6rI6_F9JfBf_2GKd8hQolA3Y/view?usp=sharing)
   - Save as `metadata.csv` in the `data/d7p/` folder

#### Rename Metadata Files
After downloading, use the helper script to rename metadata files:

```bash
python rename_metadata.py
```

**Note**: 
- The full datasets with images need to be obtained from their original sources
- The metadata files must be named `metadata.csv` in their respective directories (`data/ham10000/` and `data/d7p/`) for the scripts to work properly
- The Google Drive links provided are for metadata files only, not the complete datasets
- For detailed dataset information, see [DATASET_INFO.md](DATASET_INFO.md)

### 4. Train Models

#### LoRA Training (Recommended)
```bash
chmod +x train_lora.sh
./train_lora.sh
```

#### Full Model Training
```bash
chmod +x train_SD.sh
./train_SD.sh
```

### 5. Generate Images
```bash
# Generate a single image
python generate.py --condition "Melanoma" --mode single

# Generate a dataset
python generate.py --condition "Melanoma" --mode dataset --num_images 50
```

## 📁 Project Structure

```
LesionGen/
├── classification/          # Skin lesion classification
│   ├── classifier.py       # Main classification script
│   ├── dataloader.py       # Data loading utilities
│   ├── eval.py            # Evaluation scripts
│   └── weights/           # Pre-trained classifier weights
├── external/              # External dependencies
│   └── diffusers/         # Hugging Face Diffusers
├── data/                  # Dataset storage (downloaded separately)
│   ├── ham10000/          # HAM10000 dataset
│   └── d7p/               # D7P dataset
├── ignore/                # Excluded files and experimental code
├── train_lora.sh         # LoRA training script
├── train_SD.sh           # Full model training script
├── generate.py           # Image generation script
├── rename_metadata.py    # Metadata file renaming helper
└── requirements.txt      # Python dependencies
```

## 🔧 Training Configuration

### LoRA Training Parameters
- **Learning Rate**: 5e-06
- **Batch Size**: 1 (with gradient accumulation)
- **Max Steps**: 15,000
- **Rank**: 64
- **Resolution**: 256x256

### Full Model Training Parameters
- **Learning Rate**: 1e-05
- **Batch Size**: 1 (with gradient accumulation)
- **Max Steps**: 15,000
- **EMA**: Enabled
- **Mixed Precision**: FP16

## 🎨 Generation Options

### Single Image Generation
```bash
python generate.py --condition "Basal cell carcinoma" --mode single --output_dir my_output
```

### Dataset Generation
```bash
python generate.py --condition "Melanoma" --mode dataset --num_images 100 --output_dir synthetic_dataset
```

### Available Conditions
- Melanoma
- Basal cell carcinoma
- Benign keratosis-like lesions
- Dermatofibroma
- Melanocytic nevi
- Vascular lesions
- Actinic keratoses and intraepithelial carcinoma

## 🧪 Classification

Train and evaluate skin lesion classifiers:

```bash
cd classification
python classifier.py --data_dir /path/to/dataset --epochs 20 --batch_size 64
```

### Classification Features
- Support for ResNet18 and Vision Transformer
- Automatic data balancing
- Cross-validation support
- Model checkpointing

## 📊 Datasets

### HAM10000
- 10,015 dermatoscopic images
- 7 classes of skin lesions

### D7P
- Additional dermatological dataset


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the training framework
- [HAM10000 Dataset](https://www.nature.com/articles/sdata2018161) for dermatological images
- [Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement) for the base model


## 🔬 Citation

If you use this work in your research, please cite:

```bibtex
@article{fayyad2025lesiongen,
  title={LesionGen: A concept-guided diffusion model for dermatology image synthesis},
  author={Fayyad, Jamil and Bayasi, Nourhan and Yu, Ziyang and Najjaran, Homayoun},
  journal={arXiv preprint arXiv:2507.23001},
  year={2025}
}
```