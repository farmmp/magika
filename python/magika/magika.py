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
    # Increased from 512 to 1024 — tail bytes are often more distinctive
    # (e.g. ZIP end-of-central-directory record, PDF %%EOF marker)
    BSIZE_END: int = 1024

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
        """Identify 