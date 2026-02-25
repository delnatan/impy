"""Layer and LayerList — the core layer abstraction.

A Layer bundles a data model, visual renderer, and undo stack.
LayerList is an ordered container of layers, one per viewer window.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator

from .commands import UndoStack


@dataclass
class Layer:
    """A single annotation layer.

    Parameters
    ----------
    name : str
        Display name (must be unique within a LayerList).
    layer_type : str
        One of "shapes", "points", "tracks", "labels".
    data : Any
        The data model (ShapeData, PointData, TrackData, SparseLabels).
    visual : Any
        The vispy visual renderer for this layer.
    visible : bool
        Whether the layer is currently rendered.
    style : dict
        Type-specific style overrides (colors, line widths, etc.).
    selected_ids : set[int]
        Currently selected item IDs within this layer.
    undo_stack : UndoStack
        Per-layer undo/redo history.
    """

    name: str
    layer_type: str
    data: Any
    visual: Any
    visible: bool = True
    style: dict = field(default_factory=dict)
    selected_ids: set[int] = field(default_factory=set)
    undo_stack: UndoStack = field(default_factory=UndoStack)

    def set_visible(self, visible: bool) -> None:
        self.visible = visible
        if self.visual is not None:
            self.visual.visible = visible


VALID_LAYER_TYPES = {"shapes", "points", "tracks", "labels"}


class LayerList:
    """Ordered collection of named layers.

    One LayerList per viewer window. Replaces the separate
    window.rois, _point_layers, _track_layers, _mask_layers dicts.
    """

    def __init__(self) -> None:
        self._layers: OrderedDict[str, Layer] = OrderedDict()
        self._active: dict[str, str | None] = {t: None for t in VALID_LAYER_TYPES}

    def add(self, layer: Layer) -> None:
        """Add a layer. Name must be unique."""
        if layer.name in self._layers:
            raise ValueError(f"Layer name '{layer.name}' already exists")
        if layer.layer_type not in VALID_LAYER_TYPES:
            raise ValueError(
                f"Invalid layer type '{layer.layer_type}', "
                f"must be one of {VALID_LAYER_TYPES}"
            )
        self._layers[layer.name] = layer
        if self._active[layer.layer_type] is None:
            self._active[layer.layer_type] = layer.name

    def remove(self, name: str) -> Layer:
        """Remove and return a layer by name."""
        layer = self._layers.pop(name)
        if self._active.get(layer.layer_type) == name:
            remaining = [
                n for n, l in self._layers.items() if l.layer_type == layer.layer_type
            ]
            self._active[layer.layer_type] = remaining[0] if remaining else None
        return layer

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename a layer."""
        if old_name not in self._layers:
            raise KeyError(f"No layer named '{old_name}'")
        if new_name in self._layers:
            raise ValueError(f"Layer name '{new_name}' already exists")
        layer = self._layers[old_name]
        layer.name = new_name
        # Rebuild OrderedDict to preserve order
        new_layers = OrderedDict()
        for k, v in self._layers.items():
            new_layers[new_name if k == old_name else k] = v
        self._layers = new_layers
        if self._active.get(layer.layer_type) == old_name:
            self._active[layer.layer_type] = new_name

    def __getitem__(self, name: str) -> Layer:
        return self._layers[name]

    def __contains__(self, name: str) -> bool:
        return name in self._layers

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._layers.values())

    def __len__(self) -> int:
        return len(self._layers)

    def by_type(self, layer_type: str) -> list[Layer]:
        """Return all layers of a given type, in order."""
        return [l for l in self._layers.values() if l.layer_type == layer_type]

    def active(self, layer_type: str) -> Layer | None:
        """Return the active layer for a given type, or None."""
        name = self._active.get(layer_type)
        if name is not None and name in self._layers:
            return self._layers[name]
        return None

    def set_active(self, layer_type: str, name: str | None) -> None:
        """Set the active layer for a type."""
        if name is not None and name not in self._layers:
            raise KeyError(f"No layer named '{name}'")
        self._active[layer_type] = name

    def names(self) -> list[str]:
        """Return all layer names in order."""
        return list(self._layers.keys())
