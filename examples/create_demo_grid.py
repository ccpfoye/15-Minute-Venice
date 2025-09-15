#!/usr/bin/env python3
"""
Create a demo RasterGrid for testing the interactive viewer.

This script creates a simple test grid with various terrain types
arranged in a pattern for demonstration purposes.
"""

import numpy as np
import sys
from pathlib import Path
from affine import Affine

# Add src to path to import urbanflow
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from urbanflow import RasterGrid


def create_demo_grid(width: int = 200, height: int = 150) -> RasterGrid:
    """
    Create a demo RasterGrid with various terrain patterns.

    Parameters
    ----------
    width : int
        Grid width in cells
    height : int
        Grid height in cells

    Returns
    -------
    RasterGrid
        Demo grid ready for visualization
    """
    # Create basic RasterGrid
    raster_grid = RasterGrid(coordinate_reference_system="EPSG:4326", cell_size=1.0)

    # Define legend
    legend = {
        "ocean": 0,
        "street": 1,
        "building": 2,
        "canal": 3,
        "courtyard": 4
    }
    raster_grid.legend = legend

    # Create grid filled with ocean
    grid = np.full((height, width), legend["ocean"], dtype=np.uint8)

    # Create some patterns

    # Main streets (horizontal and vertical)
    street_spacing = 20
    for i in range(0, height, street_spacing):
        if i < height:
            grid[i, :] = legend["street"]
    for j in range(0, width, street_spacing):
        if j < width:
            grid[:, j] = legend["street"]

    # Buildings in grid pattern
    for i in range(2, height - 2, street_spacing):
        for j in range(2, width - 2, street_spacing):
            # Create building blocks
            if i + 15 < height and j + 15 < width:
                grid[i:i+15, j:j+15] = legend["building"]
                # Add courtyards in centers of some buildings
                if (i // street_spacing + j // street_spacing) % 3 == 0:
                    grid[i+5:i+10, j+5:j+10] = legend["courtyard"]

    # Add some canals
    # Diagonal canal
    for i in range(min(height, width)):
        if i < height and i < width:
            grid[i, i] = legend["canal"]
            if i + 1 < width:
                grid[i, i + 1] = legend["canal"]

    # Horizontal canal
    canal_y = height // 3
    if canal_y + 2 < height:
        grid[canal_y:canal_y+3, :] = legend["canal"]

    # Vertical canal
    canal_x = width // 2
    if canal_x + 1 < width:
        grid[:, canal_x:canal_x+2] = legend["canal"]

    # Set the grid
    raster_grid.grid = grid

    # Create a simple transform (identity transform for demo)
    raster_grid.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, height)

    return raster_grid


def main():
    """Create and save a demo grid."""
    import argparse

    parser = argparse.ArgumentParser(description="Create a demo RasterGrid")
    parser.add_argument("--output", "-o", default="demo_grid.npz", help="Output file name")
    parser.add_argument("--width", type=int, default=200, help="Grid width")
    parser.add_argument("--height", type=int, default=150, help="Grid height")

    args = parser.parse_args()

    print(f"Creating demo grid ({args.width}x{args.height})...")
    demo_grid = create_demo_grid(args.width, args.height)

    print(f"Saving to {args.output}...")
    demo_grid.save(args.output)

    print("Demo grid created! Legend:")
    for terrain, value in demo_grid.legend.items():
        print(f"  {terrain}: {value}")

    print(f"\nTo view the grid, run:")
    print(f"python examples/interactive_tcod_viewer.py {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())