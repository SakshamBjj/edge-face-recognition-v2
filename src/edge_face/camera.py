"""
Cross-platform camera initialization with fallback logic.

Handles:
- WSL2 detection and clear error messaging
- Multi-index fallback (try 0, 1, 2 if config index fails)
- Platform-specific camera availability checks
"""

import cv2
import sys
import platform


def _is_wsl() -> bool:
    """Detect if running under WSL (any version)."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def _get_platform_info() -> str:
    """Get human-readable platform description."""
    system = platform.system()
    if system == "Linux" and _is_wsl():
        return "WSL (Windows Subsystem for Linux)"
    return system


def open_camera(camera_id: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """
    Open camera with cross-platform fallback logic.

    Attempts to open the camera at the specified index. If that fails,
    tries fallback indices (0, 1, 2) before raising an error.

    Args:
        camera_id: Preferred camera index from config (default: 0)
        width: Frame width to set (default: 640)
        height: Frame height to set (default: 480)

    Returns:
        Opened cv2.VideoCapture object

    Raises:
        RuntimeError: If no camera can be opened on any index

    Note:
        WSL2 users will see a specific error message directing them to
        run on native Windows/Linux due to USB passthrough limitations.
    """
    platform_name = _get_platform_info()

    # WSL2 camera access is known to fail — warn immediately
    if _is_wsl():
        print(
            f"\n[WARNING] Detected {platform_name}\n"
            "Webcam access via WSL2 is unreliable due to USB passthrough limitations.\n"
            "Recommended: Run on native Windows or Linux.\n"
            "Attempting camera access anyway...\n",
            file=sys.stderr
        )

    # Try the configured camera_id first, then common fallback indices
    indices_to_try = [camera_id] + [i for i in [0, 1, 2] if i != camera_id]

    for idx in indices_to_try:
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            # Verify we can actually read a frame (some drivers claim isOpened() but can't read)
            ret, _ = cam.read()
            if ret:
                # Success — set resolution and return
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                if idx != camera_id:
                    print(f"[INFO] Camera index {camera_id} failed, using fallback index {idx}")
                else:
                    print(f"[INFO] Camera opened successfully (index {idx})")
                
                return cam
            cam.release()

    # All indices failed — provide platform-specific guidance
    error_msg = f"[ERROR] Cannot access webcam on {platform_name}\n\n"

    if _is_wsl():
        error_msg += (
            "WSL2 USB passthrough for webcams is not reliable.\n"
            "Solution: Run this command on native Windows:\n"
            "  1. Open Windows Terminal (not WSL)\n"
            "  2. Navigate to your project directory\n"
            "  3. Run: edge-face collect / edge-face run\n\n"
            "The package works the same on Windows — only camera I/O requires native execution.\n"
        )
    elif platform_name == "Linux":
        error_msg += (
            "Possible causes:\n"
            "  - No camera connected\n"
            "  - Camera in use by another application\n"
            "  - Permissions issue (try: sudo usermod -a -G video $USER)\n"
            f"  - Tried indices: {indices_to_try}\n\n"
            "Debug: Run 'ls /dev/video*' to see available camera devices\n"
        )
    elif platform_name == "Darwin":  # macOS
        error_msg += (
            "Possible causes:\n"
            "  - No camera connected\n"
            "  - Camera permission not granted\n"
            "  - Camera in use by another application\n"
            f"  - Tried indices: {indices_to_try}\n\n"
            "Check: System Preferences → Security & Privacy → Camera\n"
        )
    else:  # Windows or other
        error_msg += (
            "Possible causes:\n"
            "  - No camera connected\n"
            "  - Camera in use by another application (close Zoom, Teams, etc.)\n"
            "  - Driver issue\n"
            f"  - Tried indices: {indices_to_try}\n\n"
            "Debug: Open Windows Camera app to verify camera works\n"
        )

    raise RuntimeError(error_msg)


def get_available_cameras(max_check: int = 5) -> list[int]:
    """
    Detect available camera indices (for debugging/troubleshooting).

    Args:
        max_check: Maximum camera index to check (default: 5)

    Returns:
        List of available camera indices

    Example:
        >>> available = get_available_cameras()
        >>> print(f"Available cameras: {available}")
        Available cameras: [0, 2]
    """
    available = []
    for i in range(max_check):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            ret, _ = cam.read()
            if ret:
                available.append(i)
        cam.release()
    return available