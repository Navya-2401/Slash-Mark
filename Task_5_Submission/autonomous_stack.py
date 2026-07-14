import numpy as np

# --- Compatibility Fix for Modern Numpy ---
np.int = int
np.float = float
np.bool = bool
# ------------------------------------------

import cv2
import pickle
import scipy.ndimage.measurements
import collections

# ==========================================
# IMPORT PROJECT 5 HELPERS
# ==========================================
try:
    from functions_detection import draw_boxes, compute_heatmap_from_detections
    from functions_feat_extraction import find_cars
except ImportError as e:
    print(f"THE REAL ERROR IS: {e}")
    exit()

# ==========================================
# PHASE 4: PID CONTROL SIMULATION
# ==========================================
class PIDController:
    """A simple Python PID controller to simulate steering adjustments."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def update_error(self, cte):
        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte

    def total_error(self):
        return -self.kp * self.p_error - self.ki * self.i_error - self.kd * self.d_error

# ==========================================
# PHASE 2: LANE DETECTION
# ==========================================
class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = np.float32(x1), np.float32(y1)
        self.x2, self.y2 = np.float32(x2), np.float32(y2)
        self.slope = self.compute_slope()
        self.bias = self.compute_bias()

    def compute_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def compute_bias(self):
        return self.y1 - self.slope * self.x1

    def draw(self, img, color=[255, 0, 0], thickness=10):
        cv2.line(img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), color, thickness)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    ignore_mask_color = 255 if len(img.shape) == 2 else (255,) * img.shape[2]
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def compute_lane_from_candidates(line_candidates, img_shape):
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]
    
    if not pos_lines or not neg_lines:
        return None, None

    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.median([l.slope for l in neg_lines])
    left_lane = Line(0, neg_bias, -np.int32(np.round(neg_bias / neg_slope)), 0)

    lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    right_lane = Line(0, lane_right_bias, np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0])

    return left_lane, right_lane

def process_lane_lines(color_image):
    img_h, img_w = color_image.shape[0], color_image.shape[1]
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)
    
    # Define the polygon that covers just the road
    vertices = np.array([[(50, img_h), (450, 310), (490, 310), (img_w - 50, img_h)]], dtype=np.int32)
    masked_edges = region_of_interest(img_edge, vertices)

    detected_lines_raw = hough_lines_detection(masked_edges, 2, np.pi / 180, 1, 15, 5)
    line_img = np.zeros_like(color_image)
    
    if detected_lines_raw is not None:
        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines_raw]
        candidate_lines = [line for line in detected_lines if 0.5 <= np.abs(line.slope) <= 2]
        
        if candidate_lines:
            left_lane, right_lane = compute_lane_from_candidates(candidate_lines, img_gray.shape)
            if left_lane: left_lane.draw(line_img)
            if right_lane: right_lane.draw(line_img)
            
    # --- THE FIX ---
    # Apply the exact same mask to the giant drawn lines so they get cropped at the horizon
    masked_line_img = region_of_interest(line_img, vertices)
            
    # Blend the cropped lines onto the real video
    return cv2.addWeighted(color_image, 0.8, masked_line_img, 1., 0.)

# ==========================================
# PHASE 3: PERCEPTION (HOG + SVM)
# ==========================================
print("Loading pre-trained SVM model and parameters for vehicle detection...")

# --- Compatibility Fix for Older scikit-learn Pickles ---
import sys
import sklearn.svm._classes
import sklearn.preprocessing._data
sys.modules['sklearn.svm.classes'] = sklearn.svm._classes
sys.modules['sklearn.preprocessing.data'] = sklearn.preprocessing._data
# --------------------------------------------------------

try:
    svc = pickle.load(open('data/svm_trained.pickle', 'rb'))
    feature_scaler = pickle.load(open('data/feature_scaler.pickle', 'rb'))
    feat_extraction_params = pickle.load(open('data/feat_extraction_params.pickle', 'rb'))
except Exception as e:
    print(f"THE REAL ERROR IS: {e}")
    exit()

time_window = 5
hot_windows_history = collections.deque(maxlen=time_window)

def detect_vehicles(frame):
    hot_windows = []
    
    # Adjusted y_start and y_stop for the 960x540 resized frame
    for subsample in np.arange(1, 2.5, 0.5):
        hot_windows += find_cars(frame, 300, 530, subsample, svc, feature_scaler, feat_extraction_params)

    if hot_windows:
        hot_windows_history.append(hot_windows)
        hot_windows = np.concatenate(hot_windows_history)

    thresh = (time_window - 1) 
    heatmap, heatmap_thresh = compute_heatmap_from_detections(frame, hot_windows, threshold=thresh, verbose=False)

    labeled_frame, num_objects = scipy.ndimage.measurements.label(heatmap_thresh)

    final_img = frame.copy()
    for car_number in range(1, num_objects + 1):
        rows, cols = np.where(labeled_frame == car_number)
        x_min, y_min = np.min(cols), np.min(rows)
        x_max, y_max = np.max(cols), np.max(rows)
        cv2.rectangle(final_img, (x_min, y_min), (x_max, y_max), color=(0, 255, 255), thickness=4)

    return final_img

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == '__main__':
    steering_pid = PIDController(0.1, 0.001, 3.0)

    # Place a test video named 'test.mp4' inside the project_5 folder
    video_path = 'test.mp4' 
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}. Please place an mp4 video in the folder.")
        exit()

    print("Starting Autonomous Stack. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize to 960x540 to optimize processing speed and match lane metrics
        frame = cv2.resize(frame, (960, 540))

        # 1. Detect Lanes
        lane_frame = process_lane_lines(frame)

        # 2. Detect Vehicles 
        final_frame = detect_vehicles(lane_frame)

        # 3. Simulate Control 
        simulated_cte = np.random.uniform(-0.5, 0.5) 
        steering_pid.update_error(simulated_cte)
        steering_angle = steering_pid.total_error()
        
        cv2.putText(final_frame, f"Simulated Steering: {steering_angle:.3f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Task 5 - Autonomous Stack', final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()