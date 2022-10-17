import os
import sys
from os import fspath
from typing import Union

import cv2
import luigi
from luigi.util import inherits

from try_object_detection.detection import DetectionResults
from try_object_detection.detection.mobile_net import load_model
from try_object_detection.utils.types import Seq, Opt
from try_object_detection.utils.ui import window_closed
from try_object_detection.workflow import PreparedData, PrepareData
from try_object_detection.workflow.common import ConstTarget


@inherits(PrepareData)
class DetectFile(luigi.Task):
    """Detect objects in file."""

    file: str = luigi.PathParameter(description="File to be processed", exists=True)
    out: str = luigi.Parameter(default='', description="Optional output path")
    conf: float = luigi.FloatParameter(default=0.6, description="Confidence threshold.")
    results: Opt[DetectionResults] = None

    def requires(self):
        return self.clone(PrepareData)

    def output(self):
        if not self.out:
            return ConstTarget(exists=False)
        return luigi.LocalTarget(self.out)

    def run(self):
        prepared = self.prepared_data
        model = load_model(prepared.model_file, prepared.config_path)
        image = cv2.imread(fspath(self.file))
        results = DetectionResults.get(model.detect(image, confThreshold=self.conf), prepared.labels)
        results.draw(image)
        if self.out:
            os.makedirs(os.path.dirname(self.out), exist_ok=True)
            cv2.imwrite(self.out, image)
        else:
            cv2.imshow("Main", image)
            cv2.waitKeyEx()
        self.results = results

    @property
    def prepared_data(self) -> PreparedData:
        return self.input().result


@inherits(PrepareData)
class CaptureTask(luigi.Task):
    """Capture video from the given source."""

    source: Union[int, str] = luigi.Parameter(default=0, description="Capture source.")
    conf: float = luigi.FloatParameter(default=0.6, description="Confidence threshold.")

    def requires(self):
        return self.clone(PrepareData)

    def output(self):
        return ConstTarget(exists=False)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        prepared = self.prepared_data
        model = load_model(prepared.model_file, prepared.config_path)
        while True:
            ret, frame = cap.read()
            results = DetectionResults.get(model.detect(frame, confThreshold=self.conf), prepared.labels)
            results.draw(frame)
            cv2.imshow("Main", frame)
            if window_closed("Main"):
                break
        cap.release()
        cv2.destroyAllWindows()

    @property
    def prepared_data(self) -> PreparedData:
        return self.input().result


def detect_file(file: str, out: str = '', folder: str = "./data") -> DetectionResults:
    """Detect objects in a single image file."""
    detect = DetectFile(file=file, out=out, folder=folder)
    luigi.build([detect], local_scheduler=True, workers=1)
    return detect.results


def run_file(args: Seq[str] = tuple(sys.argv)):
    """Run object detection in image file using command-line arguments."""
    file_path = args[1]
    out_path = args[2] if len(args) > 2 else ''
    detect_file(file_path, out_path)


def run_capture(source: Union[str, int] = 0, conf: float = 0.6, folder: str = "./data"):
    """Run online object detection on captured frames."""
    capture = CaptureTask(source=source, conf=conf, folder=folder)
    luigi.build([capture], local_scheduler=True, workers=1)


def run_capture_cli(args: Seq[str] = tuple(sys.argv)):
    """Run online object detection using the command-line arguments."""
    source = args[1] if len(args) > 1 else 0
    conf = float(args[2]) if len(args) > 2 else 0.6
    run_capture(source, conf)


if __name__ == '__main__':
    run_capture_cli()
