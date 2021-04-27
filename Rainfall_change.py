import scipy.stats
import numpy as np ; import numpy.ma as ma
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from netCDF4 import Dataset
import netCDF4
import time
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import sys, glob, os, re

def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
    The path (and filename, without the extension) to save the
    figure to.
    ext : string (default='png')
    The file extension. This must be supported by the active
    matplotlib backend (see matplotlib.backends module).  Most
    backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
    Whether to close the figure after saving.  If you want to save
    the figure multiple times (e.g., to multiple formats), you
    should NOT close it in between saves or you will have to
    re-plot it.
    verbose : boolean (default=True)
    whether to print information about when and where the image
    has been saved.
    """
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
       directory = '.'
    #If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # The final path to save to
    savepath = os.path.join(directory, filename)
    if verbose:
        print("Saving figure to '%s'..." % savepath),
    # Actually save the figure
    plt.savefig(savepath)
    # Close it
    if close:
        plt.close()
    if verbose:
        print("Done")

path = '/home/mousta-climate/Desktop/AIMS_Lectures/data/seasons'

##########################################################################################

ncfile0 = Dataset(path + '/afr_cru_djf.nc', 'r', format='NETCDF4')
#pr_jjas = ncfile0.variables['pr'][:,:,:]*86400 # MULTIPLY BY 86400 TO CONVERT TO MM/DAY
pr_jjas = ncfile0.variables['pre'][:,:,:]
lat = ncfile0.variables['lat'][:]
lon = ncfile0.variables['lon'][:]
ncfile0.close()

pr_ref = np.mean(pr_jjas,axis=0)  

fig = plt.figure(figsize=(7.15,6.10))
kwargs = {'format': '%.0f'}  # to fix decimals at X numbers after - put **kwargs in plt.cbar 

#[lon2d, lat2d] = np.meshgrid(lon, lat)

prj = ccrs.PlateCarree(central_longitude=0.0)

axa = plt.subplot(111, projection=prj)
#axa.add_feature(cfeat.COASTLINE ,edgecolor = 'k')
axa.add_feature(cfeat.BORDERS.with_scale('10m'),linewidth=0.5)
axa.coastlines(resolution='10m',linewidth=0.5);
#axa.add_feature(cfeat.BORDERS, linestyle='-', alpha=.5)
#axa.add_feature(cfeat.OCEAN,edgecolor='k',facecolor='w') # to mask ocean
cs1 = plt.contourf(lon,lat,pr_ref,levels = np.linspace(0., 17.,11),cmap=plt.cm.gist_ncar)

axa.set_extent([-25 ,60, -40, 35]) 
axa.set_xticks(range(-25,60,15), crs=prj)
axa.set_yticks(range(-40,40,15), crs=prj)
axa.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
axa.yaxis.set_major_formatter(LATITUDE_FORMATTER)
plt.title('a) JJAS Pre. change, NF RCP26 ', fontsize=8)
plt.ylabel('')
cb0 = plt.colorbar ( cs1, ax = axa,orientation ='vertical' )
#levels = np.linspace(0., 10.,11.)


save('/home/mousta-climate/Desktop/AIMS_Lectures/figures/rainfall_changes_JJAS', ext='ps', close=True, verbose=True)  




