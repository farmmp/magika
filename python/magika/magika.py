# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core Magika class for content-type detection using deep learning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from magika.types import MagikaResult, MagikaOutputFields, ModelFeatures
from magika.content_types import ContentTypeLabel


class Magika:
    """Main interface for Magika file-type detection.

    Uses a deep learning model to identify the content type of files
    based on their byte content, independent of file extensions.

    Example:
        >>> m = Magika()
        >>> result = m.identify_path(Path("example.py"))
        >>> print(result.output.ct_label)
        'python'
    """

    # Number of bytes sampled from the beginning of the file
    BSIZE_START: int = 512
    # Number of bytes sampled from the middle of the file
    BSIZE_MIDDLE: int = 512
    # Number of bytes sampled from the end of the file
    BSIZE_END: int = 512

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        prediction_mode: str = "high-confidence",
        no_dereference: bool = False,
    ) -> None:
        """Initialize Magika with optional custom model directory.

        Args:
            model_dir: Path to a directory containing model assets.
                       Defaults to the bundled model.
            prediction_mode: One of 'best-guess', 'high-confidence', or
                             'medium-confidence'. Controls when the model
                             prediction is returned vs. a fallback label.
            no_dereference: If True, do not follow symbolic links.
        """
        self._model_dir = model_dir or self._get_default_model_dir()
        self._prediction_mode = prediction_mode
        self._no_dereference = no_dereference
        self._model = None  # Lazy-loaded on first inference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify_path(self, path: Path) -> MagikaResult:
        """Identify the content type of a single file path.

        Args:
            path: Filesystem path to the file to inspect.

        Returns:
            A MagikaResult describing the detected content type.
        """
        return self.identify_paths([path])[0]

    def identify_paths(self, paths: List[Path]) -> List[MagikaResult]:
        """Identify the content types of multiple file paths.

        Args:
            paths: List of filesystem paths to inspect.

        Returns:
            A list of MagikaResult objects, one per input path,
            in the same order as the input list.
        """
        results: List[MagikaResult] = []
        for path in paths:
            try:
                result = self._identify_single_path(path)
            except Exception as exc:  # pragma: no cover
                result = MagikaResult(
                    path=path,
                    output=MagikaOutputFields(
                        ct_label=ContentTypeLabel.UNKNOWN,
                        score=0.0,
                        group="unknown",
                        mime_type="application/octet-stream",
                        magic="",
                        description=f"Error during identification: {exc}",
                    ),
                )
            results.append(result)
        return results

    def identify_bytes(self, content: bytes) -> MagikaResult:
        """Identify the content type from raw bytes.

        Args:
            content: Raw byte string to classify.

        Returns:
            A MagikaResult with path set to None.
        """
        features = self._extract_features_from_bytes(content)
        return self._run_inference(path=None, features=features)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _identify_single_path(self, path: Path) -> MagikaResult:
        """Run identification for a single resolved path."""
        resolved = path if self._no_dereference else path.resolve()

        if not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if resolved.is_dir():
            return MagikaResult(
                path=path,
                output=MagikaOutputFields(
                    ct_label=ContentTypeLabel.DIRECTORY,
                    score=1.0,
                    group="special",
                    mime_type="inode/directory",
                    magic="directory",
                    description="A directory",
                ),
            )

        content = resolved.read_bytes()
        features = self._extract_features_from_bytes(content)
        return self._run_inference(path=path, features=features)

    def _extract_features_from_bytes(self, content: bytes) -> ModelFeatures:
        """Extract fixed-size byte features used as model input."""
        size = len(content)
        beg = list(content[: self.BSIZE_START])
        mid_start = max(0, size // 2 - self.BSIZE_MIDDLE // 2)
        mid = list(content[mid_start : mid_start + self.BSIZE_MIDDLE])
        end = list(content[max(0, size - self.BSIZE_END) :])

        # Pad sequences to fixed length with -1 sentinel
        beg += [-1] * (self.BSIZE_START - len(beg))
        mid += [-1] * (self.BSIZE_MIDDLE - len(mid))
        end += [-1] * (self.BSIZE_END - len(end))

        return ModelFeatures(beg=beg, mid=mid, end=end, size=size)

    def _run_inference(self, path, features: ModelFeatures) -> MagikaResult:
        """Load model if needed and run a forward pass."""
        if self._model is None:
            self._load_model()
        # Actual inference delegated to the loaded model wrapper
        return self._model.predict(path=path, features=features, mode=self._prediction_mode)

    def _load_model(self) -> None:
        """Lazily load the ONNX/TFLite model from disk."""
        from magika.model import MagikaModel

        self._model = MagikaModel(model_dir=self._model_dir)

    @staticmethod
    def _get_default_model_dir() -> Path:
        """Return the path to the bundled model assets."""
        return Path(__file__).parent / "models" / "standard_v1"
