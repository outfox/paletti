"""paletti -- apply colour palettes to images.

A command-line port of the palette/dither shaders from the ``palette-shader-2``
Godot project to NumPy + Pillow.
"""

from .cli import main

__all__ = ["main"]
__version__ = "0.1.0"
