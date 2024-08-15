import numpy as np
from shapely import geometry
import geopandas as gpd
import pandas as pd
from rasterio import features, transform
import fipy as fp
import h5py
from scipy import ndimage, optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections
import freqent.freqentn as fen

def get_boundary(gdf):
    """
    return GeoDataFrame with just boundary of a given GeoDataFrame
    """
    boundary = gpd.GeoDataFrame([gdf.unary_union])
    boundary.geometry = boundary[0]
    boundary.crs = gdf.crs

    return boundary


def make_boundary(data, x_grid, y_grid, crs):
    """
    Creates a geopandas boundary polygon for a numpy array,
    where the boundary separates nan from non-nan data
    
    Input
    -----
    data : array-like
    x_grid : array-like
        2D x-coordinates of data
    y_grid : array-like
        2D y-coordinates of data
    crs : str
        coordinate reference system
    """

    # get affine transformation from array indices to physical coordinates
    x_dil = (x_grid.max() - x_grid.min()) / len(np.unique(x_grid))
    y_dil = (y_grid.max() - y_grid.min()) / len(np.unique(y_grid))
    x_trans = x_grid.min()
    y_trans = y_grid.min()
    # assume no shear
    x_shear = 0
    y_shear = 0

    affine = transform.Affine(x_dil, x_shear, x_trans,
                              y_shear, y_dil, y_trans)
    shapes = features.shapes((1 - np.isnan(data)).astype(np.uint8), transform=affine)
    polygons = [geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes if shape[1] == 1]

    grid_boundary = gpd.GeoDataFrame([polygons[0]], geometry=[polygons[0]])
    # grid_boundary.geometry = grid_boundary[0]
    grid_boundary.crs = crs
    
    return grid_boundary


def make_simple_boundary(data, x_grid, y_grid, crs,
                         buffer, simplify):
    """
    Creates a "simple" geopandas boundary polygon for a numpy array,
    where the boundary separates nan from non-nan data
    
    Input
    -----
    data : array-like
    x_grid : array-like
        2D x-coordinates of data
    y_grid : array-like
        2D y-coordinates of data
    crs : str
        coordinate reference system
    buffer : float
        distance to inflate edges outwards
    simplify : float
        maximum distance that simple boundary lines can be from edges
    """
    simple_boundary = make_boundary(data, x_grid, y_grid, crs)
    
    simple_boundary.geometry = simple_boundary.geometry.buffer(buffer)
    simple_boundary["dissolve_column"] = 0
    simple_boundary = simple_boundary.dissolve(by="dissolve_column")
    simple_boundary.geometry = simple_boundary.geometry.simplify(simplify)
    
    return simple_boundary


def get_coords(boundary):
    """
    Get (x,y) coordinates of boundary, to be used to make mesh
    """
    if boundary.boundary[0].geom_type == "MultiLineString":
        longest = np.argmax([g.length for g in boundary.boundary[0].geoms])
        xx, yy = boundary.boundary[0].geoms[longest].coords.xy
    else:
        xx, yy = boundary.boundary[0].coords.xy
        
    x = np.array(xx[:-1])
    y = np.array(yy[:-1])
    return x, y


def make_mesh(data, x_grid, y_grid, crs,
              buffer, simplify, cellsize):
    """
    Start from data, create grid around non-nan values
    """
    
    simple_boundary = make_simple_boundary(data, x_grid, y_grid,
                                           crs, buffer, simplify)

    x, y = get_coords(simple_boundary)
    
    points = [f'Point({idx+1}) = {{{round(x)}, {round(y)}, 0.0, {cellsize}}};' 
              for idx, (x, y) in enumerate(zip(x, y))]
    lines = [f'Line({idx}) = {{{idx}, {((idx) % (len(points))) + 1}}};'
             for idx in range(1, len(points)+1)]
    loop_list = ', '.join([f"{idx+1}" for idx in range(len(lines))])
    loop = [f'Curve Loop(1) = {{{loop_list}}};']
    surface = ["Plane Surface(1) = {1};"]
        
    geo_file_contents = '\n'.join(np.concatenate([points, lines, 
                                                  loop, surface]))
    mesh = fp.Gmsh2D(geo_file_contents)

    return mesh, simple_boundary, geo_file_contents


def get_data(file, year=1990, region="all", norm=True, method="wb"):
    ykey = str(year)
    
    if (region == "all") | (region == "masked"):
        region_str = "masked"
    elif region == "county":
        region_str = "county"

    with h5py.File(file, "r") as d:
        x_grid = d[ykey]["x_grid"][()]
        y_grid = d[ykey]["y_grid"][()]
        white = d[ykey]["white_grid_" + region_str][()]
        black = d[ykey]["black_grid_" + region_str][()]

        if norm:
            capacity = get_capacity(file, region=region, method=method)
            ϕW = white / (1.1 * capacity)
            ϕB = black / (1.1 * capacity)
        else:
            ϕW = white
            ϕB = black

    return ϕW, ϕB, x_grid, y_grid


def get_capacity(file, region="all", method="wb"):
    with h5py.File(file, "r") as d:
        x_grid = d[list(d.keys())[0]]["x_grid"][()]
        capacity = np.zeros(x_grid.shape)

        regions = ["all", "county"]
        if region not in regions:
            raise ValueError("region is either all or county")
        else:
            if region == "all":
                region_str = "masked"
            elif region == "county":
                region_str = "county"
        
        methods = ["wb", "total"]
        if method not in methods:
            raise ValueError("method is either wb or total")

        for key in d.keys():
            if method.lower() == "wb":
                wb = (d[key]["white_grid_" + region_str][()] +
                      d[key]["black_grid_" + region_str][()])
                capacity = np.fmax(capacity, wb)
            elif method.lower() == "total":
                tot = d[key]["total_grid_" + region_str][()]
                capacity = np.fmax(capacity, tot)

        return capacity


def nansmooth(arr, sigma=1):
    """
    Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.

    stolen from: https://stackoverflow.com/a/61481246
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr

    return gauss


def expDecay(x, A, ξ, b):
    return A * np.exp(-x/ξ) + b


def get_corrLength(datafile, region="masked",
                   capacity_method="local",
                   p0=[1, 10, 0]):
    # extract data
    arr = []
    for year in [1980, 1990, 2000, 2010, 2020]:
        ϕW, _, x, _ = get_data(datafile, year, region=region,
                                capacity_method=capacity_method)
    arr.append(ϕW)
    arr = np.array(arr)

    # if nans, set to 0
    arr[np.isnan(arr)] = 0.0

    # calculate correlation function
    f, _ = fen.csdn(arr, arr, sample_spacing=[x[1, 1] - x[0, 0]])
    c = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(f.mean(axis=0))))
    # perform azimuthal average
    cr, r = fen._azimuthal_average(c, mask="circle")
    # r = sample_spacing
    popt, pcov = optimize.curve_fit(expDecay, r, cr / cr[0], p0)
    corr_length = popt[1]
    corr_length_std = np.sqrt(pcov[1, 1])

    return corr_length, corr_length_std


def dump(datafile, group_name, datadict):
    with h5py.File(datafile, "a") as d:
        grp = d.create_group(group_name)
        for key, value in datadict.items():
            grp.create_dataset(key, data=value)


def plot_mesh(data, mesh, ax,
              cmap=plt.cm.viridis,
              vmin=None, vmax=None):
    
    xmin, ymin = mesh.extents["min"]
    xmax, ymax = mesh.extents["max"]
    
    col = collections.PolyCollection(np.moveaxis(mesh.vertexCoords[:, mesh._orderedCellVertexIDs],
                                                 (0, 1, 2), (2, 1, 0)))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    col.set_color(cmap(norm(data)))
    ax.add_collection(col)
    ax.set(xlim=[xmin, xmax],
           ylim=[ymin, ymax])
    ax.set_aspect(1)