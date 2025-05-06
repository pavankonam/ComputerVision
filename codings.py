import cv2
import numpy as np
import os
from tqdm import tqdm


image_folder = 'image_girl'
out_vid = 'combined_comparison.mp4'
search_radius = 20 


im_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
num_frames = len(im_files)

# Manual initialization
first_frame = cv2.imread(os.path.join(image_folder, im_files[0]))
init_bbox = cv2.selectROI("Select Head in First Frame", first_frame, False, False)
cv2.destroyAllWindows()
x, y, w, h = map(int, init_bbox)
temp = first_frame[y:y+h, x:x+w]

# Matching function
def match_temp(frame, temp, method, prev_bbox, search_radius):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    th, tw = gray_temp.shape

    px, py = prev_bbox
    best_score = -np.inf if method != 'ssd' else np.inf
    best_loc = (px, py)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            tx = px + dx
            ty = py + dy
            if ty < 0 or tx < 0 or ty + h > gray_frame.shape[0] or tx + w > gray_frame.shape[1]:
                continue
            window = gray_frame[ty:ty + h, tx:tx + w]
            if window.shape != gray_temp.shape:
                continue

            if method == 'ssd':
                score = -np.sum((gray_temp.astype(np.float32) - window.astype(np.float32)) ** 2)
            elif method == 'cc':
                score = np.sum(gray_temp * window)
            elif method == 'ncc':
                temp_mean = np.mean(gray_temp)
                win_mean = np.mean(window)
                numerator = np.sum((gray_temp - temp_mean) * (window - win_mean))
                denominator = np.sqrt(np.sum((gray_temp - temp_mean)**2) * np.sum((window - win_mean)**2))
                score = numerator / denominator if denominator != 0 else -1

            if (method != 'ssd' and score > best_score) or (method == 'ssd' and score < best_score):
                best_score = score
                best_loc = (tx, ty)

    return best_loc


methods = ['ssd', 'cc', 'ncc']
labels = ['SSD', 'CC', 'NCC']
trackers = [{'bbox': (x, y), 'temp': temp.copy()} for _ in methods]

# Preparing output video
height, width, _ = first_frame.shape
frame_size = (width * 3, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_vid, fourcc, 20, frame_size)


print("Tracking and generating combined video...")
for i in tqdm(range(num_frames)):
    frame = cv2.imread(os.path.join(image_folder, im_files[i]))
    combined_frame_parts = []

    for m_idx, method in enumerate(methods):
        trk = trackers[m_idx]
        if i == 0:
            tx, ty = trk['bbox']
        else:
            tx, ty = match_temp(frame, trk['temp'], method, trk['bbox'], search_radius)
            trk['temp'] = frame[ty:ty+h, tx:tx+w]
            trk['bbox'] = (tx, ty)

        temp_frame = frame.copy()
        cv2.rectangle(temp_frame, (tx, ty), (tx + w, ty + h), (0, 255, 0), 2)
        cv2.putText(temp_frame, labels[m_idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        combined_frame_parts.append(temp_frame)

    # Combining side by side
    combined_frame = np.hstack(combined_frame_parts)
    out.write(combined_frame)

out.release()
print(f"\n Combined comparison video saved as '{out_vid}'")
