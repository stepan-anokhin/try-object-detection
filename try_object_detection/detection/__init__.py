from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, Tuple

import cv2
import numpy as np

from try_object_detection.utils.types import Seq

# Short name for DetectionModel results (classes, confs, bboxes)
RawResults: TypeAlias = Tuple[np.ndarray, np.ndarray, np.ndarray]

# Type hint for OpenCV2 color spec
Color: TypeAlias = Tuple[int, int, int]

DEFAULT_FONT: int = cv2.FONT_HERSHEY_PLAIN


@dataclass
class DetectionResults:
    """Object detection results."""
    classes: Seq[int]
    confidence: Seq[float]
    bboxes: Seq[Seq[int]]
    labels: Seq[str]

    @staticmethod
    def get(raw_results: RawResults, labels: Seq[str]) -> DetectionResults:
        """Get results from the DetectionModel output."""
        return DetectionResults(*raw_results, labels=labels)

    def draw(
            self,
            image: np.ndarray,
            text_size: int = 1,
            text_font: int = DEFAULT_FONT,
            text_color: Color = (0, 255, 0),
            text_thickness: int = 1,
            border_width: int = 2,
            border_color: Color = (255, 0, 0)
    ) -> np.ndarray:
        """Draw bounding boxes on the image."""
        for cls, conf, box in zip(self.classes, self.confidence, self.bboxes):
            cv2.rectangle(image, box, border_color, border_width)
            cv2.putText(
                image, self.labels[cls - 1],
                (box[0] + 10, box[1] + 10),
                text_font,
                fontScale=text_size,
                color=text_color,
                thickness=text_thickness
            )
        return image
