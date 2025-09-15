#!/usr/bin/env python3
"""
Console Rendering Demo
======================

Demonstrates how to use the tcod console rendering features in RasterGrid
and RasterGridWithPOIs classes.

This example shows:
1. Basic grid rendering with default character mappings
2. Custom character and color schemes
3. POI visualization
4. Path rendering with box-drawing characters
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import tcod

from urbanflow.RasterGrid import RasterGrid
from urbanflow.RasterGridWithPOIs import RasterGridWithPOIs


def create_demo_data():
    """
    Create a small demo raster grid and POIs for demonstration.
    """
    # Create a simple 20x20 grid for demo
    grid = np.zeros((20, 20), dtype=np.uint8)
    
    # Add some buildings (value 2)
    grid[5:8, 5:8] = 2
    grid[12:15, 12:15] = 2
    grid[3:6, 15:18] = 2
    
    # Add streets (value 1)
    grid[9, :] = 1  # horizontal street
    grid[:, 9] = 1  # vertical street
    
    # Add canals (value 3)
    grid[2, 3:17] = 3
    grid[17, 3:17] = 3
    
    # Add courtyards (value 4)
    grid[6, 6] = 4
    grid[13, 13] = 4
    
    # Create a simple transform (identity for demo)
    from affine import Affine
    transform = Affine.identity()
    
    # Create legend
    legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    
    # Create RasterGrid object
    raster_grid = RasterGrid()
    raster_grid.grid = grid
    raster_grid.transform = transform
    raster_grid.legend = legend
    raster_grid.cell_size = 1
    raster_grid.coordinate_reference_system = "EPSG:32633"
    
    # Create demo POIs
    pois_data = [
        {"uid": "POI_A", "name": "Restaurant", "geometry": Point(6, 6)},
        {"uid": "POI_B", "name": "Shop", "geometry": Point(13, 13)},
        {"uid": "POI_C", "name": "Hotel", "geometry": Point(16, 4)},
    ]
    poi_gdf = gpd.GeoDataFrame(pois_data, crs="EPSG:32633")
    
    return raster_grid, poi_gdf


def demo_basic_rendering():
    """
    Demonstrate basic console rendering with default settings.
    """
    print("=== Basic Console Rendering ===")
    
    raster_grid, _ = create_demo_data()
    
    # Basic rendering with default characters
    console = raster_grid.to_console()
    
    print("Grid rendered with default characters:")
    print("Ocean: ~, Street: ., Building: #, Canal: ≈, Courtyard: (space)")
    print()
    
    # Print a small portion to terminal
    for r in range(min(10, console.height)):
        line = ""
        for c in range(min(20, console.width)):
            line += chr(console.ch[c, r])
        print(line)
    print()


def demo_custom_characters():
    """
    Demonstrate custom character and color mappings.
    """
    print("=== Custom Character Mapping ===")
    
    raster_grid, _ = create_demo_data()
    
    # Custom character mapping
    custom_chars = {
        'ocean': '.',
        'street': '=',
        'building': '█',
        'canal': '≈',
        'courtyard': 'o'
    }
    
    # Custom color mapping
    custom_colors = {
        'ocean': ((50, 50, 100), (0, 0, 50)),      # Dark blue
        'street': ((200, 200, 200), (100, 100, 100)),  # Light/dark grey
        'building': ((200, 50, 50), (150, 0, 0)),       # Red
        'canal': ((0, 150, 200), (0, 50, 100)),         # Blue
        'courtyard': ((150, 200, 150), (50, 100, 50)),  # Green
    }
    
    console = raster_grid.to_console(char_map=custom_chars, color_map=custom_colors)
    
    print("Grid rendered with custom characters:")
    print("Ocean: ., Street: =, Building: █, Canal: ≈, Courtyard: o")
    print()
    
    # Print a small portion to terminal
    for r in range(min(10, console.height)):
        line = ""
        for c in range(min(20, console.width)):
            line += chr(console.ch[c, r])
        print(line)
    print()


def demo_pois_rendering():
    """
    Demonstrate POI rendering on top of the grid.
    """
    print("=== POI Rendering ===")
    
    raster_grid, poi_gdf = create_demo_data()
    
    # Create RasterGridWithPOIs
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(
        raster_grid, poi_gdf, do_adjusted=True
    )
    
    # Render with POIs
    console = rgp.to_console(show_pois=True, poi_char='@', poi_color=(255, 255, 0))
    
    print("Grid with POI markers (@):")
    print("POIs are overlaid on the terrain grid")
    print()
    
    # Print a small portion to terminal
    for r in range(min(15, console.height)):
        line = ""
        for c in range(min(20, console.width)):
            line += chr(console.ch[c, r])
        print(line)
    print()


def demo_path_rendering():
    """
    Demonstrate path rendering with box-drawing characters.
    """
    print("=== Path Rendering ===")
    
    raster_grid, poi_gdf = create_demo_data()
    
    # Create RasterGridWithPOIs
    rgp = RasterGridWithPOIs.from_RasterGrid_and_POIs(
        raster_grid, poi_gdf, do_adjusted=True
    )
    
    # Create a sample path between two points
    path_r = np.array([6, 7, 8, 9, 10, 11, 12, 13])
    path_c = np.array([6, 6, 6, 6, 8, 10, 12, 13])
    
    # Render with path using box drawing
    console = rgp.to_console(
        show_pois=True,
        path=(path_r, path_c),
        use_box_drawing=True,
        path_color=(255, 215, 0),
    )
    
    print("Grid with POIs and path using box-drawing characters:")
    print("● = start, ◎ = end, ─│└┘┌┐ = path segments")
    print()
    
    # Print a small portion to terminal
    for r in range(min(18, console.height)):
        line = ""
        for c in range(min(20, console.width)):
            line += chr(console.ch[c, r])
        print(line)
    print()


def demo_scaling():
    """
    Demonstrate grid downsampling for large grids.
    """
    print("=== Scaling Demo ===")
    
    raster_grid, _ = create_demo_data()
    
    # Render with scale=2 (every other cell)
    console = raster_grid.to_console(scale=2)
    
    print("Grid downsampled by factor of 2:")
    print(f"Original size: {raster_grid.grid.shape}")
    print(f"Console size: {console.width} x {console.height}")
    print()
    
    # Print the scaled version
    for r in range(console.height):
        line = ""
        for c in range(console.width):
            line += chr(console.ch[c, r])
        print(line)
    print()


def main():
    """
    Run all console rendering demonstrations.
    """
    print("TCOD Console Rendering Demo")
    print("=" * 50)
    print()
    
    try:
        demo_basic_rendering()
        demo_custom_characters()
        demo_pois_rendering()
        demo_path_rendering()
        demo_scaling()
        
        print("=== Demo Complete ===")
        print()
        print("The console objects returned by to_console() can be:")
        print("1. Saved as images using tcod's built-in functionality")
        print("2. Rendered to terminal using context managers")
        print("3. Used in interactive tcod applications")
        print("4. Processed further with tcod's rendering pipeline")
        
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Make sure urbanflow and tcod are properly installed.")
    except Exception as e:
        print(f"Error: {e}")
        print("Check that the urbanflow package is in your Python path.")


if __name__ == "__main__":
    main()