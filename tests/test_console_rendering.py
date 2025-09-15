"""
Tests for console rendering functionality in RasterGrid and RasterGridWithPOIs.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from affine import Affine
import tcod

from urbanflow.RasterGrid import RasterGrid
from urbanflow.RasterGridWithPOIs import RasterGridWithPOIs


@pytest.fixture
def sample_raster_grid():
    """Create a sample RasterGrid for testing."""
    # Create a 10x10 grid
    grid = np.zeros((10, 10), dtype=np.uint8)
    
    # Add some features
    grid[2:5, 2:5] = 2  # building
    grid[0, :] = 1      # street row
    grid[:, 0] = 1      # street column
    grid[7, 3:7] = 3    # canal
    grid[3, 3] = 4      # courtyard
    
    # Create RasterGrid
    raster_grid = RasterGrid()
    raster_grid.grid = grid
    raster_grid.transform = Affine.identity()
    raster_grid.legend = {"ocean": 0, "street": 1, "building": 2, "canal": 3, "courtyard": 4}
    raster_grid.cell_size = 1
    raster_grid.coordinate_reference_system = "EPSG:32633"
    
    return raster_grid


@pytest.fixture
def sample_poi_gdf():
    """Create a sample POI GeoDataFrame for testing."""
    pois_data = [
        {"uid": "POI_1", "name": "Test POI 1", "geometry": Point(3, 3)},
        {"uid": "POI_2", "name": "Test POI 2", "geometry": Point(7, 7)},
    ]
    return gpd.GeoDataFrame(pois_data, crs="EPSG:32633")


@pytest.fixture
def sample_rgp(sample_raster_grid, sample_poi_gdf):
    """Create a sample RasterGridWithPOIs for testing."""
    return RasterGridWithPOIs.from_RasterGrid_and_POIs(
        sample_raster_grid, sample_poi_gdf, do_adjusted=False
    )


class TestRasterGridConsoleRendering:
    """Test console rendering functionality for RasterGrid."""

    def test_to_console_basic(self, sample_raster_grid):
        """Test basic console rendering with default parameters."""
        console = sample_raster_grid.to_console()
        
        assert isinstance(console, tcod.console.Console)
        assert console.width == 10
        assert console.height == 10
        
        # Check that cells have been populated
        assert console.ch.any()
        assert console.fg.any()
        assert console.bg.any()

    def test_to_console_custom_chars(self, sample_raster_grid):
        """Test console rendering with custom character mapping."""
        custom_chars = {
            'ocean': 'O',
            'street': 'S',
            'building': 'B',
            'canal': 'C',
            'courtyard': 'Y',
        }
        
        console = sample_raster_grid.to_console(char_map=custom_chars)
        
        # Check that custom characters are used
        unique_chars = set(chr(c) for c in console.ch.flat)
        expected_chars = set(custom_chars.values())
        assert expected_chars.issubset(unique_chars)

    def test_to_console_custom_colors(self, sample_raster_grid):
        """Test console rendering with custom color mapping."""
        custom_colors = {
            'ocean': ((255, 0, 0), (0, 0, 0)),
            'street': ((0, 255, 0), (0, 0, 0)),
        }
        
        console = sample_raster_grid.to_console(color_map=custom_colors)
        
        # Check that colors have been applied
        assert console.fg.any()
        assert console.bg.any()
        
        # Check specific color values (red for ocean)
        ocean_mask = sample_raster_grid.grid == 0
        if ocean_mask.any():
            ocean_positions = np.where(ocean_mask)
            for r, c in zip(ocean_positions[0], ocean_positions[1]):
                assert tuple(console.fg[c, r]) == (255, 0, 0)

    def test_to_console_scaling(self, sample_raster_grid):
        """Test console rendering with scaling."""
        console = sample_raster_grid.to_console(scale=2)
        
        assert console.width == 5  # 10 // 2
        assert console.height == 5  # 10 // 2
        assert isinstance(console, tcod.console.Console)

    def test_to_console_existing_console(self, sample_raster_grid):
        """Test rendering to an existing console."""
        existing_console = tcod.console.Console(15, 15)
        
        console = sample_raster_grid.to_console(console=existing_console)
        
        assert console is existing_console
        assert console.width == 15
        assert console.height == 15

    def test_to_console_invalid_scale(self, sample_raster_grid):
        """Test that invalid scale values raise errors."""
        with pytest.raises(ValueError):
            sample_raster_grid.to_console(scale=0)
        
        with pytest.raises(ValueError):
            sample_raster_grid.to_console(scale=-1)

    def test_create_char_grid(self, sample_raster_grid):
        """Test the _create_char_grid helper method."""
        char_grid = sample_raster_grid._create_char_grid()
        
        assert char_grid.shape == sample_raster_grid.grid.shape
        assert char_grid.dtype.kind == 'U'  # Unicode string
        
        # Check specific characters
        assert char_grid[0, 0] == '.'  # street
        assert char_grid[2, 2] == '#'  # building
        assert char_grid[7, 3] == '≈'  # canal

    def test_create_char_grid_custom_chars(self, sample_raster_grid):
        """Test _create_char_grid with custom character mapping."""
        custom_chars = {'street': 'S', 'building': 'B'}
        char_grid = sample_raster_grid._create_char_grid(char_map=custom_chars)
        
        assert char_grid[0, 0] == 'S'  # street
        assert char_grid[2, 2] == 'B'  # building


class TestRasterGridWithPOIsConsoleRendering:
    """Test console rendering functionality for RasterGridWithPOIs."""

    def test_to_console_with_pois(self, sample_rgp):
        """Test console rendering with POI overlay."""
        console = sample_rgp.to_console(show_pois=True)
        
        assert isinstance(console, tcod.console.Console)
        assert console.width == 10
        assert console.height == 10
        
        # Check that POI characters are present
        chars = [chr(c) for c in console.ch.flat]
        assert '@' in chars  # default POI character

    def test_to_console_without_pois(self, sample_rgp):
        """Test console rendering without POI overlay."""
        console = sample_rgp.to_console(show_pois=False)
        
        # Check that POI characters are not present
        chars = [chr(c) for c in console.ch.flat]
        assert '@' not in chars

    def test_to_console_custom_poi_char(self, sample_rgp):
        """Test console rendering with custom POI character."""
        console = sample_rgp.to_console(show_pois=True, poi_char='X')
        
        chars = [chr(c) for c in console.ch.flat]
        assert 'X' in chars
        assert '@' not in chars

    def test_to_console_with_path_simple(self, sample_rgp):
        """Test console rendering with simple path."""
        path_r = np.array([1, 2, 3, 4])
        path_c = np.array([1, 2, 3, 4])
        
        console = sample_rgp.to_console(
            path=(path_r, path_c),
            use_box_drawing=False,
            path_char='*'
        )
        
        chars = [chr(c) for c in console.ch.flat]
        assert '*' in chars

    def test_to_console_with_path_box_drawing(self, sample_rgp):
        """Test console rendering with box-drawing path."""
        path_r = np.array([1, 2, 3])
        path_c = np.array([1, 1, 1])  # vertical line
        
        console = sample_rgp.to_console(
            path=(path_r, path_c),
            use_box_drawing=True
        )
        
        chars = [chr(c) for c in console.ch.flat]
        # Should contain start (●), end (◎), and vertical line (│)
        assert '●' in chars
        assert '◎' in chars
        assert '│' in chars

    def test_to_console_empty_path(self, sample_rgp):
        """Test console rendering with empty path."""
        empty_path_r = np.array([])
        empty_path_c = np.array([])
        
        console = sample_rgp.to_console(path=(empty_path_r, empty_path_c))
        
        # Should not crash and should work normally
        assert isinstance(console, tcod.console.Console)

    def test_overlay_pois_on_console(self, sample_rgp):
        """Test the _overlay_pois_on_console method."""
        console = tcod.console.Console(10, 10)
        
        sample_rgp._overlay_pois_on_console(
            console, scale=1, poi_char='@', poi_color=(255, 255, 0)
        )
        
        # Check that POI positions have the correct character
        chars = [chr(c) for c in console.ch.flat]
        assert '@' in chars

    def test_overlay_path_on_console_simple(self, sample_rgp):
        """Test the _overlay_path_on_console method with simple rendering."""
        console = tcod.console.Console(10, 10)
        path_r = np.array([1, 2, 3])
        path_c = np.array([1, 2, 3])
        
        sample_rgp._overlay_path_on_console(
            console, path_r, path_c, scale=1,
            path_char='*', path_color=(255, 0, 0),
            use_box_drawing=False
        )
        
        # Check that path character is present
        chars = [chr(c) for c in console.ch.flat]
        assert '*' in chars

    def test_overlay_path_on_console_box_drawing(self, sample_rgp):
        """Test the _overlay_path_on_console method with box drawing."""
        console = tcod.console.Console(10, 10)
        path_r = np.array([1, 2, 3])
        path_c = np.array([1, 1, 1])  # vertical line
        
        sample_rgp._overlay_path_on_console(
            console, path_r, path_c, scale=1,
            path_char='*', path_color=(255, 0, 0),
            use_box_drawing=True
        )
        
        chars = [chr(c) for c in console.ch.flat]
        assert '●' in chars  # start
        assert '◎' in chars  # end
        assert '│' in chars  # vertical line

    def test_path_bounds_checking(self, sample_rgp):
        """Test that path rendering handles out-of-bounds coordinates."""
        console = tcod.console.Console(5, 5)
        # Path that goes outside console bounds
        path_r = np.array([0, 10, 20])  # row 10, 20 are out of bounds
        path_c = np.array([0, 0, 0])
        
        # Should not crash
        sample_rgp._overlay_path_on_console(
            console, path_r, path_c, scale=1,
            path_char='*', path_color=(255, 0, 0),
            use_box_drawing=False
        )
        
        # Only the in-bounds part should be rendered
        chars = [chr(c) for c in console.ch.flat]
        assert '*' in chars

    def test_poi_bounds_checking(self, sample_rgp):
        """Test that POI rendering handles out-of-bounds coordinates."""
        console = tcod.console.Console(5, 5)
        
        # Create POI data with out-of-bounds coordinates
        sample_rgp.POI_gdf = sample_rgp.POI_gdf.copy()
        sample_rgp.POI_gdf.loc[:, 'row'] = [2, 10]  # Second POI out of bounds
        sample_rgp.POI_gdf.loc[:, 'col'] = [2, 10]
        
        # Should not crash
        sample_rgp._overlay_pois_on_console(
            console, scale=1, poi_char='@', poi_color=(255, 255, 0)
        )
        
        # Only the in-bounds POI should be rendered
        chars = [chr(c) for c in console.ch.flat]
        assert '@' in chars