from io import IOBase
from typing import Iterator

MB = 10 ** 6


def io_chunks(source: IOBase, chunk_size: int = 1 * MB) -> Iterator[bytes]:
    """Iterate over data chunks."""
    chunk = source.read(chunk_size)
    while chunk:
        yield chunk
        chunk = source.read(chunk_size)
