# Example Object Detection

## Prepare

Make sure you have [poetry](https://python-poetry.org/docs/#installation) installed on your system.

Install python dependencies:

```shell
poetry install
```

Prepare data:

```shell
poetry run prepare
```

## Detect Objects in Files

To detect objects in a single file run

```shell
poetry run detect [input-image-path]
```

To save detection results

```shell
poetry run detect [input-image-path] [output-file-path]
```

### Capture input

```shell
poetry run capture
```
