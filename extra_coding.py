import cv2
import numpy as np
import os
from tqdm import tqdm


image_folder = "image_girl"
out_vid = "ncc_tracking_output.mp4"
search_radius = 60
conf_threshold = 0.5

frame_files = sorted(
    [f for f in os.listdir(image_folder) if f.endswith('.jpg')],
    key=lambda x: int(os.path.splitext(x)[0])
)

# Reading first frame
first_frame_path = os.path.join(image_folder, frame_files[0])
first_frame = cv2.imread(first_frame_path)
if first_frame is None:
    raise FileNotFoundError(f"Cannot read: {first_frame_path}")

# Manually selecting template
init_bbox = cv2.selectROI("Select Template", first_frame, fromCenter=False)
cv2.destroyAllWindows()
x, y, w, h = map(int, init_bbox)
temp = cv2.cvtColor(first_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
prev_pos = (x, y)
lost = False


h, w, _ = first_frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_vid, fourcc, 20, (w, h))

for fname in tqdm(frame_files):
    frame = cv2.imread(os.path.join(image_folder, fname))
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, tw = temp.shape
    if not lost:
        px, py = prev_pos
        x_min = max(px - search_radius, 0)
        y_min = max(py - search_radius, 0)
        x_max = min(px + tw + search_radius, gray.shape[1] - tw)
        y_max = min(py + th + search_radius, gray.shape[0] - th)
    else:
        x_min, y_min = 0, 0
        x_max, y_max = gray.shape[1] - tw, gray.shape[0] - th

    best_score = -1
    best_loc = None
    for yy in range(y_min, y_max):
        for xx in range(x_min, x_max):
            patch = gray[yy:yy+th, xx:xx+tw]
            patch_mean = np.mean(patch)
            temp_mean = np.mean(temp)
            nume = np.sum((patch - patch_mean) * (temp - temp_mean))
            deno = np.sqrt(np.sum((patch - patch_mean)**2) * np.sum((temp - temp_mean)**2))
            score = nume / deno if deno != 0 else -1
            if score > best_score:
                best_score = score
                best_loc = (xx, yy)

    # Determining if match is good
    if best_score >= conf_threshold:
        lost = False
        prev_pos = best_loc
        cv2.rectangle(frame, best_loc, (best_loc[0]+tw, best_loc[1]+th), (0, 255, 0), 2)
        cv2.putText(frame, f"Score: {best_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        lost = True
        cv2.putText(frame, "Tracking Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out.write(frame)

out.release()
print(f"\n Output video saved to: {out_vid}")
