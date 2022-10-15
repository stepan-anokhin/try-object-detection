import cv2


def load_model(model_path: str, config_path: str) -> cv2.dnn_DetectionModel:
    """Load MobileNet detection model."""
    model = cv2.dnn_DetectionModel(model_path, config_path)
    model.setInputSize((320, 320))
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    return model
