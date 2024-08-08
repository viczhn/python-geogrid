# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:44:25 2024

@author: viczhn
"""
#code to split a large (like global) landuse data into multiple slices

import xarray as xr
import numpy as np
import os
import shutil
#from osgeo import gdal,ogr,osr
from python_geogrid import geogrid, np_byteorder, np_isigned

def construct_slice(idx, bufferw=0):
    """
    Construct slice for lon, lat
    :param idx: None, int, or float
    :param bufferw: add buffer region,
    :return:
    """
    if idx is None or isinstance(idx, (int, float)):
        idx = slice(idx, idx)
    elif isinstance(idx, (tuple, list)):
        if len(idx) == 1:
            idx = slice(idx[0], idx[0])
        else:
            idx = slice(idx[0] - bufferw, idx[1] + bufferw)
    else:
        raise ValueError("Not supported index of {}".format(idx))
    return idx
def subset(ncvar, lev=None, lat=None, lon=None, bufferwidth=0):
    """
    Subset variable by given level, lat and lon point or range
    :param ncvar: xr.DataArray object
    :param lev: the first dimension
    :param lat: the second dimension
    :param lon: the third dimension
    :param bufferwidth: a buffer zone in lon and lat range
    :return:

    .. note:: ncvar must dimension 3D (lev, lat, lon) or 2D (lat, lon)
    """

    if lev is None and lat is None and lon is None:  # do not slice, return
        return ncvar
    #print(lat,lon)
    lat =  construct_slice(lat, bufferwidth)
    lon =  construct_slice(lon, bufferwidth)
    if ncvar.ndim == 3:
        lev =  construct_slice(lev)
        #logger.debug("{}  |  {}  |  {}".format(lev, lat, lon))
        return ncvar.loc[lev, lat, lon]
    elif ncvar.ndim == 2:
        #print(lat,lon)
        return ncvar.loc[lat, lon]
    else:
        raise NotImplementedError("NCVAR must be 2D or 3D.")
def map_lu(varin,missing=-9999):
    if varin in [10,11,12,20]:
        return 12
    elif varin in [30,40]:
        return 14
    elif  varin in [70,71,72]:
        return 1 #
    elif varin in [80,81,82]:
        return 3
    elif varin in [50]:
        return 2
    elif varin in [60,61,62]:
        return 4
    elif varin in [90]:
        return 5
    elif varin in [120,121]:
        return 6
    elif varin in [122]:
        return 7
    elif varin in [100]:
        return 8
    elif varin in [110]:
        return 9
    elif varin in [130]:
        return 10
    elif varin in [140]:
        return 20
    elif varin in [150,151,152,153]:
        return 16
    elif varin in [160,170,180]:
        return 11
    elif varin in [190]:
        return 13
    elif varin in [200,201,202]:
        return 19
    elif varin in [210]:
        return 17
    elif varin in [220]:
        return 15
    else:
        return missing
#using EAS LC data as example
landuse_file='./C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.nc'
output_path='./geog/EAS_2022/'

missing_value=0
#attention that the dimension of every slice of WPS static file should not be larger than 99999*99999, 
slon=30
elon=170
slat=0
elat=90
#the dimension of every slice 
xsep= 1800
ysep= 1800
#read data
ds=xr.open_dataset(landuse_file)
lu_da=ds['lccs_class'][0]
ds.close
del ds
lu_da=subset(lu_da, lat=[elat,slat], lon=[slon,elon])

lons_new,lats_new=np.meshgrid(lu_da['lon'].values,lu_da['lat'].values)
lu_new=lu_da.values

ny,nx=lu_new.shape
#######################################
#reverse variables, make sure data starts from low left coner!
if lats_new[1,0]<lats_new[0,0]:
    
    lats_new=lats_new[::-1]
    lons_new=lons_new[::-1]
    lu_new=lu_new[::-1]
###############################################################################
#################################################
output_file_fmt=os.path.join(output_path,'{istart:05d}-{iend:05d}.{jstart:05d}-{jend:05d}')

if not os.path.exists(output_path):
    os.makedirs(output_path)
            
for ii in range(0,nx,xsep):
    for jj in range(0,ny,ysep):
                
        istart=ii;iend=min((ii+xsep),nx)
        jstart=jj;jend=min((jj+ysep),ny)
        print('write ','{istart:05d}-{iend:05d}.{jstart:05d}-{jend:05d}'.format(istart=istart+1,
                                                                                iend=iend,
                                                                                jstart=jstart+1,
                                                                                jend=jend))
        output_file= output_file_fmt.format(istart=istart+1,iend=iend,
                                                                    jstart=jstart+1,jend=jend)

        ##############
        if (jend-jstart)!=ysep or (iend-istart)!=xsep:
            #
            var=np.ones((ysep,xsep),dtype=lu_new.dtype)*missing_value
            var[:(jend-jstart),:(iend-istart) ]=lu_new[jstart:jend,istart:iend ]
        else:
            var  = lu_new[jstart:jend,istart:iend ]
        var  = np.vectorize(map_lu)(var) #convert eas landcover classes to modis classes
        var=var.astype(np.uint16)
        
        lons = lons_new[jstart:jend,istart:iend ]
        lats = lats_new[jstart:jend,istart:iend ]
        
        
        ###################
        if ii==0 and jj==0:
            known_lat=lats_new[0,0]
            known_lon=lons_new[0,0]
            dx=lons_new[0,1]-lons_new[0,0]
            dy=lats_new[1,0]-lats_new[0,0]

        out = geogrid("write")
    
        out.set_index(key="type"        , value="categorical")
        out.set_index(key="signed"      , value=np_isigned(var.dtype))
        out.set_index(key="endian"      , value=np_byteorder(var.dtype))
        out.set_index(key="category_min", value=1)
        out.set_index(key="category_max", value=20)
        out.set_index(key="projection"  , value="regular_ll")
        out.set_index(key="dx"          , value=dx)
        out.set_index(key="dy"          , value= dy)
        out.set_index(key="known_x"     , value=1)
        out.set_index(key="known_y"     , value=1)
        out.set_index(key="known_lat"   , value=known_lat)
        out.set_index(key="known_lon"   , value=known_lon)
        out.set_index(key="wordsize"    , value=var.dtype.itemsize)
        out.set_index(key="tile_x"      , value=xsep)
        out.set_index(key="tile_y"      , value=ysep)
        out.set_index(key="tile_z"      , value=1)
        out.set_index(key="units"       , value="\"category\"")
        out.set_index(key="description" , value="\"EAS modified-IGBP landuse - 300 meter\"")
        out.set_index(key="mminlu"      , value="\"MODIFIED_IGBP_MODIS_NOAH\"")
        out.set_index(key="iswater"     , value=17)
        out.set_index(key="isice"       , value=15)
        out.set_index(key="isurban"     , value=13)
        out.set_index(key="missing_value" , value=missing_value)
        
        
        out.write_geogrid(var, index_root=output_path)
        
        del var,out,lons,lats
        #
        
        #python-geogrid pack can only generate one slice file, so we rename it and move it to target directory
        shutil.copyfile('./00001-{xsize:05d}.00001-{ysize:05d}'.format(xsize=xsep,ysize=ysep),output_file)
        
        os.remove('./00001-{xsize:05d}.00001-{ysize:05d}'.format(xsize=xsep,ysize=ysep))
        
del lons_new,lats_new,lu_new,lu_da
print('All done!')