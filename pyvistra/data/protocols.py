"""
Structural typing contracts for 5D image data.

These Protocols describe the *minimum* interface that any pyvistra-aware
producer or consumer of 5D image data should rely on. They exist so that
external libraries (e.g. deconlib, memsolve) can type-annotate their
inputs/outputs against pyvistra without importing concrete proxy classes
or otherwise coupling to its implementation.

All 5D data uses the canonical ``(T, Z, C, Y, X)`` axis order.
"""

from typing import Any, Callable, Protocol, Tuple, runtime_checkable


@runtime_checkable
class Readable5D(Protocol):
    """A 5D array-like object that supports slice reads.

    Implemented by every proxy in :mod:`pyvistra.data.proxies` and by
    :class:`pyvistra.data.buffer.ImageBuffer`.
    """

    shape: Tuple[int, int, int, int, int]
    dtype: Any
    ndim: int

    def __getitem__(self, key: Any) -> Any: ...


@runtime_checkable
class Writable5D(Readable5D, Protocol):
    """A :class:`Readable5D` that also accepts slice writes.

    Implemented by :class:`pyvistra.data.buffer.ImageBuffer`. Writers
    should treat any object that satisfies this Protocol as a destination
    they can stream results into.
    """

    def __setitem__(self, key: Any, value: Any) -> None: ...


@runtime_checkable
class ObservableBuffer(Writable5D, Protocol):
    """A :class:`Writable5D` that publishes change notifications.

    ``subscribe(callback)`` returns a zero-arg unsubscribe function. The
    callback receives the *key* passed to ``__setitem__`` and fires on the
    writer's thread; Qt consumers must marshal to the GUI thread (see
    :class:`pyvistra.ui.window.ImageWindow` for the canonical bridge).
    """

    def subscribe(
        self, callback: Callable[[Any], None]
    ) -> Callable[[], None]: ...
