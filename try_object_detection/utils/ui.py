import cv2

KEY_ESC = 27


def window_closed(name: str, wait: int = 10) -> bool:
    """Check window is closed."""
    if cv2.getWindowProperty(name, 0) < 0:
        return True
    key = cv2.waitKey(wait) & 0xFF
    return key == ord('q') or key == KEY_ESC
