# ==========================================
# 2. SCREENSHOT UTILS
# ==========================================
import mss
from PIL import Image

def take_screenshot(rect):
    if not rect: return None
    x, y, w, h = rect
    try:
        with mss.mss() as sct:
            monitor = {"top": y, "left": x, "width": w, "height": h}
            sct_img = sct.grab(monitor)
            return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    except Exception:
        return None
