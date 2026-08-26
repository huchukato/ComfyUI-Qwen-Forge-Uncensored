"""
{
  "name": "ComfyUI-Qwen-Uncensored",
  "description": "Clean, modular ComfyUI node pack for uncensored Qwen multimodal and text generation with pluggable backends.",
  "author": "huchukato",
  "version": "1.0.0",
  "url": "https://github.com/huchukato/ComfyUI-Qwen-Uncensored",
  "category": "video"
}
"""

import importlib.util
import os
import sys

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
WEB_DIRECTORY = "./web"

current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def _load_module(file_path: str) -> None:
    module_name = os.path.basename(file_path)[:-3]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)


nodes_dir = os.path.join(current_dir, "nodes")
if os.path.isdir(nodes_dir):
    for filename in sorted(os.listdir(nodes_dir)):
        if filename.endswith(".py") and filename != "__init__.py":
            _load_module(os.path.join(nodes_dir, filename))

NODE_CLASS_MAPPINGS = dict(
    sorted(NODE_CLASS_MAPPINGS.items(), key=lambda x: NODE_DISPLAY_NAME_MAPPINGS.get(x[0], x[0]))
)
NODE_DISPLAY_NAME_MAPPINGS = dict(sorted(NODE_DISPLAY_NAME_MAPPINGS.items(), key=lambda x: x[1]))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
