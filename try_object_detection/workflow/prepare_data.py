import os.path
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import luigi
import luigi.format

from try_object_detection.utils.io_utils import io_chunks
from try_object_detection.utils.tar import untar


class DownloadTask(luigi.Task):
    """Download file task from the web."""

    url: str = luigi.Parameter(description="Source URL of the remote file.")
    destination: str = luigi.Parameter(description="Destination file path.")
    chunk_size: int = luigi.IntParameter(default=10 ** 6, description="File write chunk size", significant=False)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.destination, format=luigi.format.Nop)

    def run(self):
        os.makedirs(os.path.dirname(self.destination), exist_ok=True)
        with urllib.request.urlopen(self.url) as remote_file, self.output().open(mode="w") as destination_file:
            for chunk in io_chunks(remote_file, self.chunk_size):
                destination_file.write(chunk)


class ExtractTarTask(luigi.Task):
    """Extract .tar file contents."""

    archive: str = luigi.PathParameter(exists=True)


class PrepareModel(luigi.Task):
    """Prepare MobileNet-SSD v3 model."""
    folder: str = luigi.Parameter(default="./data", description="Destination folder.")

    MOBNET_URL: str = "http://download.tensorflow.org/models/object_detection" \
                      "/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz"
    MOBNET_FILE: str = "ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz"

    def requires(self):
        os.makedirs(self.folder, exist_ok=True)
        yield DownloadTask(url=self.MOBNET_URL, destination=os.path.join(self.folder, self.MOBNET_FILE))

    def output(self):
        return luigi.LocalTarget(self.destination_path)

    def run(self):
        archive = self.input()[0]
        with tempfile.TemporaryDirectory() as tempdir:
            untar(archive.path, tempdir)
            shutil.move(os.path.join(tempdir, self.content_root), self.destination_path)

    @property
    def destination_path(self) -> str:
        """Calculate destination path."""
        return os.path.join(self.folder, self.content_root)

    @property
    def content_root(self) -> str:
        """Archive content root."""
        return self.MOBNET_FILE.removesuffix(".tar.gz")


@dataclass
class PreparedData:
    """Prepared data description."""
    model_path: str
    config_path: str
    labels_path: str

    @cached_property
    def labels(self) -> Sequence[str]:
        """Load labels."""
        with open(self.labels_path) as file:
            return tuple(map(str.strip, file.readlines()))


class PrepareData(luigi.Task):
    """Download MobileNet-SSD v3 model and config files."""
    folder: str = luigi.Parameter(default="./data", description="Destination folder.")

    CONFIG_URL: str = "https://gist.githubusercontent.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7" \
                      "/raw/2a20064a9d33b893dd95d2567da126d0ecd03e85/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    CONFIG_FILE: str = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

    LABELS_URL: str = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

    def requires(self):
        os.makedirs(self.folder, exist_ok=True)
        yield PrepareModel(folder=self.folder)
        yield DownloadTask(url=self.CONFIG_URL, destination=os.path.join(self.folder, self.CONFIG_FILE))
        yield DownloadTask(url=self.LABELS_URL, destination=os.path.join(self.folder, "coco.names"))

    def complete(self):
        """Always should check for requirements."""
        return False

    @property
    def result(self) -> PreparedData:
        """Get prepared data results."""
        model, config, labels = self.input()
        return PreparedData(model_path=model.path, config_path=config.path, labels_path=labels.path)


def run() -> PreparedData:
    """Download required data."""
    prepare = PrepareData()
    luigi.build([prepare], local_scheduler=True, workers=1)
    return prepare.result


if __name__ == '__main__':
    print(run())
