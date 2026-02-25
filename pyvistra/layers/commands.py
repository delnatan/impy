"""Command pattern for undo/redo support.

Each command encapsulates a reversible mutation on a data model.
Commands are stored in an UndoStack that belongs to each Layer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Command(Protocol):
    """Protocol for undoable commands.

    Commands operate on a data object (ShapeData, PointData, etc.).
    Each command must be able to execute itself and undo itself.
    """

    def execute(self, data: Any) -> None: ...
    def undo(self, data: Any) -> None: ...


class UndoStack:
    """Per-layer undo/redo stack.

    Stores executed commands and supports undo/redo navigation.
    Pushing a new command clears the redo stack.
    """

    def __init__(self, max_size: int = 100):
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._max_size = max_size

    def push(self, cmd: Command, data: Any) -> None:
        """Execute a command and push it onto the undo stack."""
        cmd.execute(data)
        self._undo.append(cmd)
        self._redo.clear()
        if len(self._undo) > self._max_size:
            self._undo.pop(0)

    def undo(self, data: Any) -> bool:
        """Undo the last command. Returns True if an undo was performed."""
        if not self._undo:
            return False
        cmd = self._undo.pop()
        cmd.undo(data)
        self._redo.append(cmd)
        return True

    def redo(self, data: Any) -> bool:
        """Redo the last undone command. Returns True if a redo was performed."""
        if not self._redo:
            return False
        cmd = self._redo.pop()
        cmd.execute(data)
        self._undo.append(cmd)
        return True

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
