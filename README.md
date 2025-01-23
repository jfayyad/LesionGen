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