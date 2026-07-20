"""Reader for Carl Zeiss CZI microscopy files."""

import numpy as np


class CZIReader:
    """
    A reader for Carl Zeiss CZI files using aicspylibczi.

    Attributes:
        filepath (str): Path to the .czi file.
        dtype (np.dtype): Data type of the image.
        shape (tuple): Dimensions (Time, Channels, Z, Y, X).
        voxel_size (tuple): (Z, Y, X) voxel size in microns.
        timestamps (list): List of timestamps for each time point.
        channels_info (list): List of dicts containing metadata (Name, Wavelengths).
        n_channels (int): Number of channels.
        n_timepoints (int): Number of timepoints.

    Methods:
        read(c=0, t=0, z=None, res_level=0): Return 2D/3D array.
        close(): Release resources.
    """

    def __init__(self, filepath):
        try:
            from aicspylibczi import CziFile
        except ImportError:
            raise ImportError(
                "CZI support requires aicspylibczi. "
                "Install with: pip install pyvistra[czi]"
            )

        self.filepath = filepath
        self._czi = CziFile(filepath)

        # Initialize containers
        self.voxel_size = (1.0, 1.0, 1.0)
        self.timestamps = []
        self.channels_info = []

        # Parse structure and metadata
        self._parse_dimensions()
        self._parse_metadata()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Release resources."""
        self._czi = None

    def _parse_dimensions(self):
        """Parse CZI dimensions and build shape."""
        # Get dimension string (e.g., 'TCZYX', 'STCZYX', 'CZYX')
        dims = self._czi.dims

        # Get shape for each dimension
        dims_shape = self._czi.get_dims_shape()

        # dims_shape is a list of dicts, one per scene
        # Use first scene (index 0) for dimensions
        if dims_shape:
            shape_dict = dims_shape[0]
        else:
            shape_dict = {}

        # Extract dimensions, defaulting to 1 if not present
        # CZI can have: S(Scene), T(Time), C(Channel), Z(Depth), Y, X, M(Mosaic), B(Block)
        self.n_timepoints = shape_dict.get("T", (0, 1))[1]
        self.n_channels = shape_dict.get("C", (0, 1))[1]
        self.size_z = shape_dict.get("Z", (0, 1))[1]
        self.size_y = shape_dict.get("Y", (0, 1))[1]
        self.size_x = shape_dict.get("X", (0, 1))[1]

        # Store dimension string for read operations
        self._dims = dims

        # Shape in (T, C, Z, Y, X) order to match ImarisReader
        self.shape = (
            self.n_timepoints,
            self.n_channels,
            self.size_z,
            self.size_y,
            self.size_x,
        )

        # Determine dtype from a small read
        try:
            sample = self._czi.read_image(T=0, C=0, Z=0)
            if isinstance(sample, tuple):
                sample = sample[0]  # (data, dims) tuple
            self.dtype = sample.dtype
        except Exception:
            self.dtype = np.uint16  # Common default for microscopy

    def _parse_metadata(self):
        """Parse metadata from CZI XML."""
        try:
            meta = self._czi.meta
        except Exception:
            return

        if meta is None:
            return

        # Parse voxel sizes from XML
        self._parse_voxel_sizes(meta)
        self._parse_channels(meta)

    def _parse_voxel_sizes(self, meta):
        """Extract voxel sizes from XML metadata."""
        try:
            # Look for scaling information
            scaling = meta.find(".//Scaling/Items")
            if scaling is None:
                return

            vox_x, vox_y, vox_z = 1.0, 1.0, 1.0

            for distance in scaling.findall("Distance"):
                dim_id = distance.get("Id")
                value_elem = distance.find("Value")
                if value_elem is not None and value_elem.text:
                    try:
                        # Value is in meters, convert to microns
                        value_m = float(value_elem.text)
                        value_um = value_m * 1e6
                        if dim_id == "X":
                            vox_x = value_um
                        elif dim_id == "Y":
                            vox_y = value_um
                        elif dim_id == "Z":
                            vox_z = value_um
                    except ValueError:
                        pass

            self.voxel_size = (vox_z, vox_y, vox_x)

        except Exception:
            pass

    def _parse_channels(self, meta):
        """Extract channel information from XML metadata."""
        self.channels_info = []

        try:
            channels = meta.findall(".//Channel")

            for i, channel in enumerate(channels):
                info = {
                    "id": i,
                    "name": f"Channel {i}",
                    "emission_wavelength": None,
                    "excitation_wavelength": None,
                }

                # Channel name
                name = channel.get("Name")
                if name:
                    info["name"] = name

                # Emission wavelength
                em = channel.find("EmissionWavelength")
                if em is not None and em.text:
                    try:
                        info["emission_wavelength"] = float(em.text)
                    except ValueError:
                        pass

                # Excitation wavelength
                ex = channel.find("ExcitationWavelength")
                if ex is not None and ex.text:
                    try:
                        info["excitation_wavelength"] = float(ex.text)
                    except ValueError:
                        pass

                self.channels_info.append(info)

        except Exception:
            pass

        # Ensure we have info for all channels
        while len(self.channels_info) < self.n_channels:
            idx = len(self.channels_info)
            self.channels_info.append(
                {
                    "id": idx,
                    "name": f"Channel {idx}",
                    "emission_wavelength": None,
                    "excitation_wavelength": None,
                }
            )

    def read(self, c=0, t=0, z=None, res_level=0):
        """
        Read image data.

        Args:
            c: Channel index.
            t: Timepoint index.
            z: Z-slice index. None for full volume.
            res_level: Resolution level (0=full). Note: CZI multi-resolution
                      support depends on file structure.

        Returns:
            np.ndarray: 2D array (Y, X) if z is specified, 3D (Z, Y, X) otherwise.
        """
        # Build keyword arguments for read_image
        kwargs = {"C": c, "T": t}

        # Handle extra dimensions that may be present
        if "S" in self._dims:
            kwargs["S"] = 0  # Use first scene
        if "M" in self._dims:
            kwargs["M"] = 0  # Use first mosaic tile
        if "B" in self._dims:
            kwargs["B"] = 0  # Use first block

        if z is not None:
            # Single Z slice
            kwargs["Z"] = z
            result = self._czi.read_image(**kwargs)
            if isinstance(result, tuple):
                data = result[0]
            else:
                data = result

            # Squeeze to (Y, X)
            return np.squeeze(data)
        else:
            # Full Z volume
            volume = []
            for zi in range(self.size_z):
                kwargs["Z"] = zi
                result = self._czi.read_image(**kwargs)
                if isinstance(result, tuple):
                    data = result[0]
                else:
                    data = result
                volume.append(np.squeeze(data))

            return np.array(volume)

    def __repr__(self):
        return (
            f"<CZIReader: {self.filepath}\n"
            f"  Shape (T,C,Z,Y,X): {self.shape}\n"
            f"  Dtype: {self.dtype}\n"
            f"  Voxel Size (um): {self.voxel_size}\n"
            f"  Channels: {[c['name'] for c in self.channels_info]}>"
        )


if __name__ == "__main__":
    print("CZI Reader Module Loaded.")
