#!/usr/bin/env python3
"""
Simple terminal-based viewer for RasterGrid using ANSI escape codes.

This example demonstrates how to:
1. Load a RasterGrid
2. Render it to terminal with viewport scrolling
3. Handle keyboard input for movement without external dependencies

Controls:
- Arrow keys or WASD: Move viewport
- +/-: Zoom in/out
- q: Quit
- r: Reset to origin
"""

import sys
import termios
import tty
import select
from pathlib import Path

# Add src to path to import urbanflow
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from urbanflow import RasterGrid


class SimpleGridViewer:
    def __init__(self, raster_grid: RasterGrid, window_width: int = 80, window_height: int = 24):
        """
        Initialize the simple viewer.

        Parameters
        ----------
        raster_grid : RasterGrid
            The grid to display
        window_width : int
            Console width in characters
        window_height : int
            Console height in characters (minus 3 for status and controls)
        """
        self.raster_grid = raster_grid
        self.window_width = window_width
        self.window_height = window_height - 3  # Reserve space for status

        # Viewport position (top-left corner in grid coordinates)
        self.viewport_x = 0
        self.viewport_y = 0

        # Zoom level (downsampling factor)
        self.zoom = 1

        # Grid dimensions
        self.grid_height, self.grid_width = raster_grid.grid.shape

        # Character and color mappings
        self.char_map = {
            'ocean': '~',
            'street': '.',
            'building': '#',
            'canal': '≈',
            'courtyard': ' ',
        }

        # ANSI color codes
        self.color_map = {
            'ocean': '\033[94m',      # Blue
            'street': '\033[37m',     # White
            'building': '\033[91m',   # Red
            'canal': '\033[96m',      # Cyan
            'courtyard': '\033[92m',  # Green
        }
        self.reset_color = '\033[0m'

        # Create reverse legend mapping
        self.value_to_key = {v: k for k, v in raster_grid.legend.items()}

    def clear_screen(self):
        """Clear the terminal screen."""
        print('\033[2J\033[H', end='')

    def get_char_at(self, grid_x: int, grid_y: int) -> tuple:
        """Get character and color for a grid position."""
        if 0 <= grid_y < self.grid_height and 0 <= grid_x < self.grid_width:
            # Apply zoom by sampling
            actual_y = grid_y * self.zoom
            actual_x = grid_x * self.zoom
            if actual_y < self.grid_height and actual_x < self.grid_width:
                cell_value = self.raster_grid.grid[actual_y, actual_x]
                terrain_key = self.value_to_key.get(cell_value, 'ocean')
                char = self.char_map.get(terrain_key, '?')
                color = self.color_map.get(terrain_key, self.reset_color)
                return char, color
        return ' ', self.reset_color

    def render_frame(self):
        """Render the current viewport to the terminal."""
        self.clear_screen()

        # Calculate effective grid dimensions at current zoom
        effective_width = self.grid_width // self.zoom
        effective_height = self.grid_height // self.zoom

        # Clamp viewport to valid range
        max_x = max(0, effective_width - self.window_width)
        max_y = max(0, effective_height - self.window_height)
        self.viewport_x = max(0, min(self.viewport_x, max_x))
        self.viewport_y = max(0, min(self.viewport_y, max_y))

        # Render grid
        for row in range(self.window_height):
            line = ""
            for col in range(self.window_width):
                grid_x = self.viewport_x + col
                grid_y = self.viewport_y + row
                char, color = self.get_char_at(grid_x, grid_y)
                line += f"{color}{char}{self.reset_color}"
            print(line)

        # Status line
        status = f"Pos: ({self.viewport_x},{self.viewport_y}) Zoom: {self.zoom}x Grid: {self.grid_width}x{self.grid_height}"
        print(f"\n{status}")
        print("Controls: Arrow/WASD=move, +/-=zoom, r=reset, q=quit")

    def get_key(self):
        """Get a single keypress from stdin."""
        if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

    def run(self):
        """Main loop."""
        # Set terminal to raw mode
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())

            print("Loading Venice RasterGrid Viewer...")
            print("Controls:")
            print("- Arrow keys or WASD: Move viewport")
            print("- +/-: Zoom in/out")
            print("- r: Reset to origin")
            print("- q: Quit")
            print("\nPress any key to start...")
            sys.stdin.read(1)

            while True:
                self.render_frame()

                # Handle input
                key = self.get_key()
                if key is None:
                    continue

                # Movement
                move_speed = max(1, self.window_width // 10)

                if key in ['A', '\033[A', 'w', 'W']:  # Up arrow or W
                    self.viewport_y -= move_speed
                elif key in ['B', '\033[B', 's', 'S']:  # Down arrow or S
                    self.viewport_y += move_speed
                elif key in ['D', '\033[D', 'a', 'A']:  # Left arrow or A
                    self.viewport_x -= move_speed
                elif key in ['C', '\033[C', 'd', 'D']:  # Right arrow or D
                    self.viewport_x += move_speed

                # Zoom
                elif key in ['+', '=']:
                    if self.zoom > 1:
                        old_zoom = self.zoom
                        self.zoom -= 1
                        # Adjust viewport for new zoom
                        self.viewport_x = self.viewport_x * old_zoom // self.zoom
                        self.viewport_y = self.viewport_y * old_zoom // self.zoom

                elif key == '-':
                    if self.zoom < 20:
                        old_zoom = self.zoom
                        self.zoom += 1
                        # Adjust viewport for new zoom
                        self.viewport_x = self.viewport_x * old_zoom // self.zoom
                        self.viewport_y = self.viewport_y * old_zoom // self.zoom

                # Reset
                elif key in ['r', 'R']:
                    self.viewport_x = 0
                    self.viewport_y = 0

                # Quit
                elif key in ['q', 'Q', '\x03']:  # q, Q, or Ctrl+C
                    break

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.clear_screen()
            print("Goodbye!")


def main():
    """Example usage with a sample grid file."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple terminal viewer for RasterGrid")
    parser.add_argument("grid_file", help="Path to .npz grid file")
    parser.add_argument("--width", type=int, default=80, help="Window width in characters")
    parser.add_argument("--height", type=int, default=24, help="Window height in characters")

    args = parser.parse_args()

    # Load the grid
    try:
        print(f"Loading grid from {args.grid_file}...")
        raster_grid = RasterGrid.load(args.grid_file)
        print(f"Loaded grid: {raster_grid.grid.shape} cells")
    except Exception as e:
        print(f"Error loading grid: {e}")
        return 1

    # Create and run viewer
    viewer = SimpleGridViewer(raster_grid, args.width, args.height)
    viewer.run()

    return 0


if __name__ == "__main__":
    exit(main())