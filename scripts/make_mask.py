import pandas as pd
from scipy.interpolate import griddata
from osgeo import gdal
from osgeo import osr
import os
import numpy as np
from shapely.geometry import mapping
from rasterio.enums import Resampling
import rioxarray as rxr
import geopandas as gpd
import subprocess
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Make numpy images and masks from SAR geotiffs and ice charts.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_dir', metavar='D', type=str, help='Directory where getotiff and ice chart is '
                                                                        'located.', dest='data_dir')
    parser.add_argument('-s', '--save_dir', metavar='S', type=str, default='../data/', help='Directory where numpy '
                                                                                            'images and masks are '
                                                                                            'saved.', dest='save_dir')
    parser.add_argument('-l', '--lake', metavar='L', type=str, default='erie', help='Lake for which the data is being '
                                                                                    'made for.', dest='lake')

    return parser.parse_args()


def main(data_dir: str, save_dir: str, lake: str):
    # extract date from directory name, note that this assumes the following format:
    # RS2_OK49620_PK471053_DK420584_SCWA_20140308_112238_HH_HV_SGF
    date = data_dir.split('_')[8]

    # load ice chart into dataframe
    dat_file = os.path.join(data_dir, 'ice_chart.dat')
    df = pd.read_csv(dat_file, sep="  ", header=None)
    df.columns = ["lat", "lon", "ice_con", "other1", "other2"]

    # define shapefile name and set temporary variable names
    lake_shapefile = f'/home/dsola/repos/PGA-Net/data/shapefiles/hydro_p_Lake{lake.capitalize()}.shp'
    tmp_file_name = 'myraster.tif'  # name of tif file created from ice chart
    hh_img_name = 'imagery_HH.tif'  # name of HH image with projection info
    hv_img_name = 'imagery_HV.tif'  # name of HV image with projection info

    # create contour map using ice chart
    numcols, numrows = 5000, 5000
    xi = np.linspace(df.lat.min(), df.lat.max(), numrows)
    yi = np.linspace(df.lon.min(), df.lon.max(), numcols)
    xi, yi = np.meshgrid(xi, yi)
    x, y, z = df.lat, df.lon, df.ice_con
    points = np.vstack((x, y)).T
    values = z
    wanted = (xi, yi)
    zi = griddata(points, values, wanted, method='nearest')

    # create geotiff file from ice chart contour map
    xmin, ymin, xmax, ymax = [xi.min(), yi.min(), xi.max(), yi.max()]
    nrows, ncols = np.shape(zi[::-1, :])
    xres = (xmax - xmin) / float(ncols)
    yres = (ymax - ymin) / float(nrows)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    output_raster = gdal.GetDriverByName('GTiff').Create(os.path.join(data_dir, tmp_file_name), ncols, nrows, 1,
                                                         gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    output_raster.SetProjection(srs.ExportToWkt())
    output_raster.GetRasterBand(1).WriteArray(zi[::-1, :])
    output_raster.FlushCache()
    data = output_raster.ReadAsArray()

    # add projection data to SAR geotiffs given the ground control points in the original geotiff
    new_hh_img_name, new_hv_img_name = hh_img_name.split('.')[0] + '_proj.tif', hv_img_name.split('.')[0] + '_proj.tif'
    subprocess.run(["gdalwarp", "-t_srs", "epsg:4326", os.path.join(data_dir, hh_img_name),
                    os.path.join(data_dir, new_hh_img_name)])
    subprocess.run(["gdalwarp", "-t_srs", "epsg:4326", os.path.join(data_dir, hv_img_name),
                    os.path.join(data_dir, new_hv_img_name)])

    # load SAR images, ice chart, and lake shapefile
    chart = rxr.open_rasterio(os.path.join(data_dir, tmp_file_name), masked=True).squeeze()
    sar_hh = rxr.open_rasterio(os.path.join(data_dir, new_hh_img_name), masked=True).squeeze()
    sar_hv = rxr.open_rasterio(os.path.join(data_dir, new_hv_img_name), masked=True).squeeze()
    crop_extent = gpd.read_file(lake_shapefile)
    crop_extent.set_crs("EPSG:4326")

    # clip images and ice chart to the extent of the lake
    sar_hh_clipped = sar_hh.rio.clip(crop_extent.geometry.apply(mapping))
    sar_hv_clipped = sar_hv.rio.clip(crop_extent.geometry.apply(mapping))
    chart_clipped = chart.rio.clip(crop_extent.geometry.apply(mapping))

    # reshape and re-project SAR images to match ice chart exactly
    sar_hh_reshaped = sar_hh_clipped.rio.reproject(
        sar_hh_clipped.rio.crs,
        shape=(chart_clipped.rio.height, chart_clipped.rio.width),
        resampling=Resampling.bilinear,
    )
    sar_hv_reshaped = sar_hv_clipped.rio.reproject(
        sar_hv_clipped.rio.crs,
        shape=(chart_clipped.rio.height, chart_clipped.rio.width),
        resampling=Resampling.bilinear,
    )
    sar_hh_reshaped = sar_hh_reshaped.rio.reproject_match(chart_clipped)
    sar_hv_reshaped = sar_hv_reshaped.rio.reproject_match(chart_clipped)

    # convert geotiffs to numpy data
    sar_hh_np = sar_hh_reshaped.data
    sar_hv_np = sar_hv_reshaped.data
    mask_np = chart_clipped.data
    assert sar_hh_np.shape == sar_hv_np.shape == mask_np.shape, "sar and mask are not the same shape"

    # combine SAR channels into a single array and remove edges of SAR image from image and mask
    sar_np = np.stack([sar_hh_np, sar_hv_np])
    edge = np.all((sar_np == 0), axis=0)
    edges = np.stack([edge, edge])
    sar_np = np.where(edges, np.nan, sar_np)
    mask_np = np.where(edge, np.nan, mask_np)
    img_nan = np.all((sar_np == np.nan), axis=0)
    mask_np = np.where(img_nan, np.nan, mask_np)

    # given a threshold of 0.5, split ice chart into two classes where class 0 is <0.5 and class 1 is >0.5
    is_low = (mask_np < 0.5)
    is_hi = (mask_np >= 0.5)
    mask_np = np.where(is_low, 0, mask_np)
    mask_np = np.where(is_hi, 1, mask_np)

    # set nan to -1 in image and mask so loss function can ignore these pixels during training
    sar_np = np.nan_to_num(sar_np, nan=-1)
    sar_np[sar_np > 256] = -1  # ignore unrealistically high pixels occasionally caused by rioxarray re-projection bug
    img_nan = np.all((sar_np == -1), axis=0)
    mask_np = np.where(img_nan, np.nan, mask_np)
    mask_np = np.nan_to_num(mask_np, nan=-1)

    # save numpy arrays of the sar image and ground truth mask for later training
    np.save(os.path.join(save_dir, "imgs", f"img_{date}_{lake}.npy"), sar_np)
    np.save(os.path.join(save_dir, "masks", f"img_{date}_{lake}.npy"), mask_np)


if __name__ == '__main__':
    args = get_args()
    main(args.data_dir, args.save_dir, args.lake)
