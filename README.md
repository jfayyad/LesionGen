# LesionGen

## Description


## Instructions

### Requirements

1. Install the requirements for the main project:

   ```bash
   pip install -r requirements.txt
   ```

2. Navigate to the `diffusers` submodule and install its requirements:

   ```bash
   cd external/diffusers
   pip install .
   cd ../..
   ```

### Download Dataset

To download and extract the dataset, run the following commands:

```bash
gdown --id 1TnR-r6TZvlbrTefXBTJSJbUyuNCXKf57 -O dataset.tar.xz
mkdir -p data && tar -xvf dataset.tar.xz -C data/
rm dataset.tar.xz
```

### Train Lora Diffusion

To train the Lora diffusion model, run the following commands:

```bash
chmod +x train_lora.sh
./train_lora.sh
```


### Usage

### Command-Line Arguments

- `--condition` (required): The condition or subject for the generated images (e.g., "Melanoma").
- `--mode` (required): Choose between "single" or "dataset".
  - `single`: Generates one image.
  - `dataset`: Generates a dataset of images.
- `--output_dir` (optional): Directory to save the generated images (default: `generated_images`).
- `--num_images` (optional): Number of images to generate (only for `dataset` mode, default: 10).

### Examples

#### Generate a Single Image

```bash
python generate.py --condition "Melanoma" --mode single
```

Output:

- The generated image will be saved in `generated_images/single/Melanoma.png`.

#### Generate a Dataset of Images

```bash
python generate_images.py --condition "Melanoma" --mode dataset --num_images 10
```

Output:

- 10 diverse images will be saved in `generated_images/dataset/Melanoma/` with filenames `1.png`, `2.png`, ..., `10.png`.


#### Notes

- The script requires pre-trained LoRA weights ([Download here](https://drive.google.com/file/d/1Q4Re52SvdMs0Xe2ddOqJQDgssHylWJrz/view?usp=drive_link)) located at `Lora/weights/checkpoint-15000`.