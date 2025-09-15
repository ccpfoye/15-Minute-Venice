#!/usr/bin/env python3
"""
Interactive TCOD viewer for RasterGrid with movement controls.

This example demonstrates how to:
1. Load a RasterGrid
2. Render it to a tcod console with viewport scrolling
3. Handle keyboard input for movement
4. Display the grid in a fixed window size

Controls:
- Arrow keys or WASD: Move viewport
- Q/ESC: Quit
- +/-: Zoom in/out
- Home: Reset to origin
"""

import tcod
import tcod.event
import sys
from pathlib import Path

# Add src to path to import urbanflow
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from urbanflow import RasterGrid


class InteractiveGridViewer:
    def __init__(self, raster_grid: RasterGrid, window_width: int = 80, window_height: int = 50):
        """
        Initialize the interactive viewer.

        Parameters
        ----------
        raster_grid : RasterGrid
            The grid to display
        window_width : int
            Console width in characters
        window_height : int
            Console height in characters
        """
        self.raster_grid = raster_grid
        self.window_width = window_width
        self.window_height = window_height

        # Viewport position (top-left corner in grid coordinates)
        self.viewport_x = 0
        self.viewport_y = 0

        # Zoom level (downsampling factor)
        self.zoom = 1

        # Grid dimensions
        self.grid_height, self.grid_width = raster_grid.grid.shape

        # Create main console
        self.console = tcod.console.Console(window_width, window_height, order='F')

        # Create a console for the full grid (will be rendered once and reused)
        self._full_console = None
        self._render_full_grid()

    def _render_full_grid(self):
        """Render the entire grid to a console."""
        self._full_console = self.raster_grid.to_console(scale=self.zoom)

    def _update_viewport(self):
        """Update the main console with the current viewport."""
        # Clear the console
        self.console.clear()

        # Calculate effective grid dimensions at current zoom
        effective_width = self.grid_width // self.zoom
        effective_height = self.grid_height // self.zoom

        # Clamp viewport to valid range
        max_x = max(0, effective_width - self.window_width)
        max_y = max(0, effective_height - self.window_height)
        self.viewport_x = max(0, min(self.viewport_x, max_x))
        self.viewport_y = max(0, min(self.viewport_y, max_y))

        # Copy visible region from full console to main console
        src_x = self.viewport_x
        src_y = self.viewport_y
        src_width = min(self.window_width, effective_width - src_x)
        src_height = min(self.window_height, effective_height - src_y)

        if src_width > 0 and src_height > 0:
            # Blit from full console to main console
            self._full_console.blit(
                dest=self.console,
                dest_x=0, dest_y=0,
                src_x=src_x, src_y=src_y,
                width=src_width, height=src_height
            )

        # Add status text
        status = f"Pos: ({self.viewport_x},{self.viewport_y}) Zoom: {self.zoom}x Grid: {self.grid_width}x{self.grid_height}"
        self.console.print(0, self.window_height - 1, status, fg=(255, 255, 255), bg=(0, 0, 0))

    def handle_event(self, event: tcod.event.Event) -> bool:
        """
        Handle keyboard input.

        Returns
        -------
        bool
            True to continue, False to quit
        """
        if isinstance(event, tcod.event.KeyDown):
            key = event.sym

            # Movement (arrow keys or WASD)
            move_speed = max(1, self.window_width // 10)  # Adaptive movement speed

            if key == tcod.event.KeySym.UP or key == tcod.event.KeySym.W:
                self.viewport_y -= move_speed
            elif key == tcod.event.KeySym.DOWN or key == tcod.event.KeySym.S:
                self.viewport_y += move_speed
            elif key == tcod.event.KeySym.LEFT or key == tcod.event.KeySym.A:
                self.viewport_x -= move_speed
            elif key == tcod.event.KeySym.RIGHT or key == tcod.event.KeySym.D:
                self.viewport_x += move_speed

            # Zoom controls
            elif key == tcod.event.KeySym.PLUS or key == tcod.event.KeySym.EQUALS:
                if self.zoom > 1:
                    self.zoom -= 1
                    self._render_full_grid()
                    # Adjust viewport for new zoom level
                    self.viewport_x *= 2
                    self.viewport_y *= 2

            elif key == tcod.event.KeySym.MINUS:
                if self.zoom < 10:
                    self.zoom += 1
                    self._render_full_grid()
                    # Adjust viewport for new zoom level
                    self.viewport_x //= 2
                    self.viewport_y //= 2

            # Reset to origin
            elif key == tcod.event.KeySym.HOME:
                self.viewport_x = 2500
                self.viewport_y = 1000

            # Quit
            elif key == tcod.event.KeySym.Q or key == tcod.event.KeySym.ESCAPE:
                return False

        return True

    def run(self):
        """Main game loop."""
        # Create a simple default tileset
        tileset = tcod.tileset.load_tilesheet(
            "data/Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
        )

        # Initialize tcod with proper keyword arguments
        context_args = {
            "columns": self.window_width,
            "rows": self.window_height,
            "title": "Venice RasterGrid Viewer",
        }
        if tileset is not None:
            context_args["tileset"] = tileset

        with tcod.context.new(**context_args) as context:

            print("Controls:")
            print("- Arrow keys or WASD: Move viewport")
            print("- +/-: Zoom in/out")
            print("- Home: Reset to origin")
            print("- Q/ESC: Quit")
            print("\nStarting viewer...")

            while True:
                # Update display
                self._update_viewport()
                context.present(self.console)

                # Handle events
                for event in tcod.event.wait():
                    if not self.handle_event(event):
                        return


def main():
    """Example usage with a sample grid file."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive TCOD viewer for RasterGrid")
    parser.add_argument("grid_file", help="Path to .npz grid file")
    parser.add_argument("--width", type=int, default=80, help="Window width in characters")
    parser.add_argument("--height", type=int, default=50, help="Window height in characters")

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
    viewer = InteractiveGridViewer(raster_grid, args.width, args.height)
    viewer.run()

    return 0


if __name__ == "__main__":
    exit(main())