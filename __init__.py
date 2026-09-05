"""
ki MiniMax H3 Reference to Video
Extended official MiniMaxH3ReferenceToVideo with intermediate short-edge
ref_image_size options: 1.2 / 1.4 / 1.6 / 1.8 / 2.0 (between match and max).

Node id / class: ki_MiniMaxH3ReferenceToVideo
Does NOT conflict with the built-in official node.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
