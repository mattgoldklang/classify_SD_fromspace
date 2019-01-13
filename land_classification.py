#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Opens a Satellite image with GDAL and performs Kmeans analysis 
@author: mattgoldklang
"""

import gdal
import sys
import numpy as np
from gdalconst import *
from sklearn.cluster import KMeans
import time as tm
import matplotlib.pyplot as plt

def open_image(directory):
    """
    opens a geoTIFF file
    """
    image_ds = gdal.Open(directory,GA_ReadOnly)
    if image_ds is None:
        print('No image here')
        sys.exit(1)
    return(image_ds)
    
def get_img_param(img):
    """
    creates a list of image parameters
    """
    cols = img.RasterXSize
    rows = img.RasterYSize
    num_bands = img.RasterCount
    img_gt = img.GetGeoTransform()
    img_proj = img.GetProjection()
    img_driver = img.GetDriver()
    img_params = [cols, rows, num_bands, img_gt, img_proj, img_driver]
    return(img_params)

def output_ds(out_array, img_params, bands, d_type=GDT_Unknown, fn = 'results.tif'):
    """
    returns classified image to file
    """
    cols,rows,gt,proj = img_params[0],img_params[1],img_params[3],img_params[4]
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    
    out_ras = driver.Create(fn,cols,rows,bands,d_type)
    out_ras.SetGeoTransform(gt)
    out_ras.SetProjection(proj)
    
    out_band = out_ras.GetRasterBand(1)
    out_band.WriteArray(out_array)
    out_band.SetNoDataValue(-99)
    out_band.FlushCache()
    out_band.GetStatistics(0,1)
    
    return()

def build_band_stack(img, num_bands):
    """
    runs through the different wv bands and masks for clouds, returns as single stacked array
    """
    band_list = []
    #cloud file used for masking
    cld = open_image('/Users/mattgoldklang/Downloads/example/LC08_L1TP_040037_20130810_20170309_01_T1_QB.tif')
    cld_band = cld.GetRasterBand(1)
    clds = cld_band.ReadAsArray(0,0)
    clds_loc = np.where(clds > 0)
    for band in range(num_bands):
        b = img.GetRasterBand(band+1)
        b_array = b.ReadAsArray(0,0)
        b_array[clds_loc[0],clds_loc[1]] = 10E6
        b_array = np.ma.masked_where(b_array == 10E6, b_array)
        band_list.append(b_array)
    band_stack = np.dstack(band_list)
    return(band_stack)

def main():
    """
    runs the kmeans analysis
    """
    #ESA Landsat image of San Diego County
    img_dir = '/Users/mattgoldklang/Downloads/example/example.tif'
    img = open_image(img_dir)
    img_param = get_img_param(img)
    
    band_ds = build_band_stack(img, img_param[2])
    flat_ds = band_ds.reshape(band_ds.shape[0]*band_ds.shape[1],band_ds.shape[2])
    input_ds = flat_ds.astype(np.double)
    #define the amount of land classes
    k_means = KMeans(n_clusters=5, n_jobs = -1)
    labels = k_means.fit_predict(input_ds)
    ods1 = labels.reshape(band_ds.shape[0], band_ds.shape[1])
    
    #output_ds(ods1, img_param, bands = img_param[2], d_type = GDT_Int16, fn = 'kmeans_sklearn.tif')
    plt.imshow(ods1)
    return(ods1)

output = main()
if __name__ == "__main__":
    start = tm.time()
    main()
    f = tm.time() - start
    print(f'\nProcessing time: {f} seconds')