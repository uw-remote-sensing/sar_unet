# Sar UNet

This repo has a simple training and evaluation script for using a UNet with SAR data.  

## How to Use

* Images shold be placed in `data/imgs/` and ground truth should be placed in `data/masks/`.
  * Note that the name for an image and its associated mask should be the exact same. E.g. `data/imgs/img1.npy` and `data/masks/img1.npy`
* `data/txt_files/train.txt` and `data/txt_files/test.txt` should contain the names of the traing and testing images respectively, with each image name on a new line. Example below:


      img1.npy
      img2.npy
      img3.npy

* Install requirements: `pip install -r requirements.txt`.
* Install package lakeseg package: `pip install -e .`.
* Run training script: `python lakeseg/train.py -d path/to/data/ -ch 2 -cl 2`.
  * Note that the above command assumes two ice classes (ice and water) and two SAR channels (HH and HV).
  * Training info can be found at the wandb link produced by the script.
  * Checkpoint are saved in temporary `lakeseg/checkpoints/` directory.

## Generating Data

* The `scripts/make_mask.py` script creates UNet compatible masks from `ice_chat.dat` files.
  * `ice_chart.dat` files contain lat/lon points with an associated sea ice concentration.
  * This script creates a contour mask from the lat/lon points, converts the contour map into a geotiff, adds projection info to a SAR geotiff with ground control points (GCPs), clips both geotiffs using a lake shapefile (found in `data/shapefiles/`), and saved both the SAR image and mask as a `.npy` files.
