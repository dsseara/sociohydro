import numpy as np
from shapely import geometry
import geopandas as gpd
import pandas as pd
from rasterio import features, transform
import fipy as fp


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

    grid_boundary = gpd.GeoDataFrame([polygons[0]])
    grid_boundary.geometry = grid_boundary[0]
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

