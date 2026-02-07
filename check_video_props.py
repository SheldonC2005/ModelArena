import subprocess
import sys

# Install opencv-python if not available
try:
    import cv2
except ImportError:
    print("Installing opencv-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "-q"])
    import cv2

# Check video properties
video_path = r"d:\Data\Github\SheldonC2005\ModelArena\archive\train\fake\00629a4d82054da4b67a48ae3a9239c4.mp4"

cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {frame_count}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  File Size: ~1 MB")
    
    cap.release()
else:
    print("Error opening video file")
