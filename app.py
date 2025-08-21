import os
from datetime import datetime
import atexit
import csv
import time
import base64
from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import threading
from datetime import datetime
import numpy as np

# Try to import OpenCV, use mock if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("WARNING: OpenCV (cv2) is not installed. Using mock camera.")
    CV2_AVAILABLE = False

# Try to import face recognition libraries
try:
    import torch
    from PIL import Image
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("WARNING: Face recognition libraries not available. Limited functionality.")
    FACE_RECOGNITION_AVAILABLE = False
    # Create dummy classes for the case when imports fail
    class MTCNN:
        def __init__(self, **kwargs):
            pass
        
        def detect(self, *args, **kwargs):
            return None, None
        
        def __call__(self, *args, **kwargs):
            return None
    
    class InceptionResnetV1:
        def __init__(self, **kwargs):
            pass
        
        def eval(self):
            return self
        
        def to(self, device):
            return self
        
        def __call__(self, *args, **kwargs):
            return torch.tensor([]) if 'torch' in globals() else np.array([])

# Try to import InsightFace for higher-accuracy embeddings (optional)
try:
    import insightface  # type: ignore
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False

# Try to import PIL independently for image encoding when OpenCV is unavailable
try:
    from PIL import Image as PILImage  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    PILImage = None  # type: ignore

app = Flask(__name__)
app.secret_key = 'face_attendance_secret_key'

# Initialize FaceNet components only if available
if FACE_RECOGNITION_AVAILABLE:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.7]
    )

    # Initialize FaceNet for face recognition
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # Initialize InsightFace analysis app if available
    if INSIGHTFACE_AVAILABLE:
        try:
            insight_app = insightface.app.FaceAnalysis(name='buffalo_l')
            # ctx_id=-1 forces CPU; set 0 to use first GPU if available
            insight_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
            print('InsightFace initialized')
        except Exception as e:
            print(f"WARNING: Failed to initialize InsightFace: {e}")
            INSIGHTFACE_AVAILABLE = False
else:
    # Create dummy device and models
    device = None
    mtcnn = MTCNN()
    facenet = InceptionResnetV1()

# Context processor to make 'now' available to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Student Database class
class StudentDatabase:
    def __init__(self):
        self.students = {}
        self.embeddings = np.array([])
        self.names = []
        self.roll_numbers = []
        self.load_database()
    
    def load_database(self):
        """Load student database from files"""
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        database_file = os.path.join(data_dir, 'students.csv')
        embeddings_file = os.path.join(data_dir, 'embeddings.npy')
        
        if os.path.exists(database_file):
            with open(database_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        roll_no, name, embedding_path = row
                        self.students[roll_no] = {
                            'name': name,
                            'embedding_path': embedding_path
                        }
        
        if os.path.exists(embeddings_file) and FACE_RECOGNITION_AVAILABLE:
            try:
                data = np.load(embeddings_file, allow_pickle=True)
                if len(data) > 0:
                    self.embeddings = data['embeddings']
                    self.names = data['names'].tolist()
                    self.roll_numbers = data['roll_numbers'].tolist()
            except Exception as e:
                print(f"Error loading embeddings: {e}")
    
    def save_database(self):
        """Save student database to files"""
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        database_file = os.path.join(data_dir, 'students.csv')
        embeddings_file = os.path.join(data_dir, 'embeddings.npy')
        
        # Save student info
        with open(database_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for roll_no, info in self.students.items():
                writer.writerow([roll_no, info['name'], info['embedding_path']])
        
        # Save embeddings only if we have face recognition available
        if FACE_RECOGNITION_AVAILABLE and len(self.embeddings) > 0:
            np.savez(embeddings_file, 
                    embeddings=self.embeddings,
                    names=np.array(self.names),
                    roll_numbers=np.array(self.roll_numbers))
    
    def add_student(self, roll_no, name, embedding):
        """Add a new student to the database"""
        if not FACE_RECOGNITION_AVAILABLE:
            print("Face recognition not available. Cannot add student.")
            return
            
        data_dir = 'data/embeddings'
        os.makedirs(data_dir, exist_ok=True)
        
        embedding_path = os.path.join(data_dir, f'{roll_no}.npy')
        np.save(embedding_path, embedding)
        
        self.students[roll_no] = {
            'name': name,
            'embedding_path': embedding_path
        }
        
        # Update arrays
        if len(self.embeddings) == 0:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        self.names.append(name)
        self.roll_numbers.append(roll_no)
        
        self.save_database()
    
    def recognize_face(self, embedding, threshold=0.6):
        """Recognize a face using cosine similarity"""
        if not FACE_RECOGNITION_AVAILABLE or len(self.embeddings) == 0:
            return None, None, float('inf')
        
        # Calculate cosine similarity
        denom = np.linalg.norm(embedding) or 1.0
        embedding = embedding / denom
        db_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        db_norms[db_norms == 0] = 1.0
        db_embeddings = self.embeddings / db_norms
        
        similarities = np.dot(db_embeddings, embedding)
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > threshold:
            return (self.roll_numbers[best_match_idx], 
                   self.names[best_match_idx], 
                   best_similarity)
        
        return None, None, best_similarity

# Initialize student database
student_db = StudentDatabase()

# Global variables
camera = None
attendance_mode = 'lab'  # Default mode: 'lab' or 'class'
attendance_marked = False  # Flag to track when attendance is marked
streaming_enabled = False  # Controls whether the backend should stream frames

# Sound notification is now handled by JavaScript in the browser
def play_success_sound():
    # Sound is now played by the browser using JavaScript
    pass

# Ensure directories exist
def ensure_directories():
    os.makedirs('static/sounds', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('attendance', exist_ok=True)
    os.makedirs('faces', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/embeddings', exist_ok=True)

def create_simple_error_frame():
    """Create a simple error frame without OpenCV"""
    # This is a minimal JPEG representation for error cases
    return (b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00'
            b'\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19'
            b'\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\'\x27\x27\x27\x25\x1c\x1c\x1c\x1c\x1c'
            b'\x1c\x1c\x1c\x1c\x1c\x1c\x1c\x1c\x1c\x1c\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01'
            b'\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\t\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x92\x00\xff\xd9')


def encode_frame_to_jpeg(frame_array):
    """Encode a numpy frame to JPEG bytes using OpenCV if available, else Pillow, else a static JPEG."""
    # Try OpenCV first
    try:
        if CV2_AVAILABLE:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', frame_array, encode_param)
            if ret:
                return buffer.tobytes()
    except Exception as e:
        print(f"OpenCV JPEG encode failed: {e}")
    # Fallback to Pillow
    if PIL_AVAILABLE and PILImage is not None:
        try:
            from io import BytesIO
            if isinstance(frame_array, bytes):
                return frame_array
            if hasattr(frame_array, 'dtype') and frame_array.dtype != np.uint8:
                frame_array = frame_array.astype(np.uint8)
            # Convert BGR->RGB if it looks like a color image
            if hasattr(frame_array, 'ndim') and frame_array.ndim == 3 and frame_array.shape[2] in (3, 4):
                # Assume BGR input for compatibility with OpenCV-like arrays
                rgb_array = frame_array[:, :, :3][:, :, ::-1]
                img = PILImage.fromarray(rgb_array)
            else:
                img = PILImage.fromarray(frame_array)
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=80)
            return buf.getvalue()
        except Exception as e:
            print(f"Pillow JPEG encode failed: {e}")
    # Last resort: return a minimal static JPEG
    return create_simple_error_frame()



# Mock camera for when no physical camera is available
class MockCamera:
    def __init__(self):
        self.frame_count = 0
        self.width = 640
        self.height = 480
        self.is_open = True
        
    def isOpened(self):
        return self.is_open
        
    def read(self):
        # Create a blank frame with a message
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add a message about the mock camera
        if CV2_AVAILABLE:
            cv2.putText(frame, "Mock Camera Active - No Real Camera Found", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "This is a simulation for testing", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add a moving element to show it's active
            self.frame_count += 1
            position = (50 + (self.frame_count % 400), 200)
            cv2.circle(frame, position, 20, (0, 165, 255), -1)
        else:
            # Simple frame generation without OpenCV
            # Create a pattern that changes over time
            self.frame_count += 1
            for y in range(self.height):
                for x in range(self.width):
                    # Create a simple pattern
                    if (x + y + self.frame_count) % 20 < 10:
                        frame[y, x] = [0, 0, 255]  # Red
                    else:
                        frame[y, x] = [0, 0, 0]    # Black
            
            # Add a moving element
            center_x = 50 + (self.frame_count % 400)
            center_y = 200
            radius = 20
            for y in range(max(0, center_y - radius), min(self.height, center_y + radius)):
                for x in range(max(0, center_x - radius), min(self.width, center_x + radius)):
                    if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                        frame[y, x] = [0, 165, 255]  # Orange
        
        return True, frame
        
    def release(self):
        self.is_open = False
        print("Mock camera released")
        
    def set(self, prop_id, value):
        # Mock implementation of set property
        if CV2_AVAILABLE:
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                self.width = int(value)
            elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = int(value)
        else:
            # Handle property IDs without OpenCV
            if prop_id == 3:  # CAP_PROP_FRAME_WIDTH
                self.width = int(value)
            elif prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
                self.height = int(value)
        return True

def get_mock_camera():
    print("Initializing mock camera")
    return MockCamera()

# Camera handling with fallback mechanism
# Replace the get_camera() function with this improved version:
def get_camera():
    global camera
    if camera is not None and hasattr(camera, 'isOpened') and camera.isOpened():
        return camera
        
    # If OpenCV is not available, always use mock camera
    if not CV2_AVAILABLE:
        print("OpenCV not available. Using mock camera.")
        camera = get_mock_camera()
        return camera
        
    try:
        # Release any existing camera first
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None
            
        # Try to open the camera with multiple attempts
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                camera = cv2.VideoCapture(0)
                if camera.isOpened():
                    # Set camera properties for better performance
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    print(f"Camera opened successfully on attempt {attempt+1}")
                    return camera
                else:
                    print(f"Attempt {attempt+1}/{max_attempts}: Could not open camera.")
                    if camera is not None:
                        camera.release()
                    time.sleep(1)  # Wait before retrying
            except Exception as e:
                print(f"Camera attempt {attempt+1} failed: {e}")
                if camera is not None:
                    camera.release()
                camera = None
                time.sleep(1)
        
        print("Error: Failed to open camera after multiple attempts. Using mock camera.")
        camera = get_mock_camera()
        return camera
        
    except Exception as e:
        print(f"Error initializing camera: {e}. Using mock camera.")
        camera = get_mock_camera()
        return camera

def release_camera():
    global camera
    if camera is not None:
        try:
            camera.release()
            print("Camera released successfully")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        finally:
            camera = None

# Generate frames for video streaming - Optimized implementation with FaceNet
def generate_frames():
    global attendance_marked, camera, streaming_enabled
    
    # Ensure camera is initialized
    if camera is None:
        camera = get_camera()
        
    # Create or load attendance file for today
    today = datetime.now().strftime('%Y-%m-%d')
    attendance_file = os.path.join('attendance', f'{attendance_mode}_{today}.csv')
    
    # Initialize attendance record if not exists
    os.makedirs('attendance', exist_ok=True)
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['RollNo', 'Name', 'Time', 'Mode'])
    
    # Load already marked attendance
    marked_attendance = {}
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row and len(row) >= 3:
                    roll_no = row[0]
                    mode = row[3] if len(row) > 3 else 'unknown'
                    if roll_no not in marked_attendance:
                        marked_attendance[roll_no] = []
                    marked_attendance[roll_no].append(mode)
    
    # Check if face recognition is available
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition not available. Running in limited functionality mode.")
    if not CV2_AVAILABLE:
        print("OpenCV not available. Using mock camera with limited functionality.")
    
    # Get camera
    cap = get_camera()
    if cap is None:
        # If camera can't be opened, return an error message
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if CV2_AVAILABLE:
            cv2.putText(error_frame, "Camera not available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
        else:
            # Simple error frame without OpenCV
            # Make sure we only call this when we have the function defined
            frame = create_simple_error_frame()
        
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)
    
    # Performance optimization and tracking variables
    frame_count = 0
    process_every_n_frames = 3  # Only process every 3rd frame for face detection
    recognition_cooldown = {}   # Cooldown timer for each detected face position
    cooldown_frames = 30        # Number of frames to wait before re-recognizing a face
    tracked_faces = {}          # face_id -> {bbox, color, label, status, ttl, last_update}
    tracked_ttl_frames = 10
    smooth_alpha = 0.6

    def compute_iou(box_a, box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0
    
    while True:
        # If streaming has been disabled, stop the generator
        if not streaming_enabled:
            # Release camera if we're the only stream
            release_camera()
            break
        try:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera")
                # Instead of breaking, try to reinitialize the camera
                release_camera()
                time.sleep(1)  # Wait a bit before trying to reconnect
                cap = get_camera()
                if cap is None:
                    # If camera still can't be opened, show error frame
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    if CV2_AVAILABLE:
                        cv2.putText(error_frame, "Camera not available", (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        ret, buffer = cv2.imencode('.jpg', error_frame)
                        frame = buffer.tobytes()
                    else:
                        # Create a simple error frame without OpenCV
                        frame = create_simple_error_frame()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    continue
                # Try to read a frame from the reinitialized camera
                success, frame = cap.read()
                if not success:
                    # If still failing, break the loop
                    print("Failed to read frame after camera reinitialization")
                    break
            
            frame_count += 1
            
            # Add mode display
            if CV2_AVAILABLE:
                cv2.putText(frame, f"Mode: {attendance_mode.upper()}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # If face recognition is not available, just display the frame with a message
            if not FACE_RECOGNITION_AVAILABLE and CV2_AVAILABLE:
                cv2.putText(frame, "Face recognition not available", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Running in limited mode", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Only process every n frames to improve performance if face recognition is available
            if FACE_RECOGNITION_AVAILABLE and CV2_AVAILABLE and frame_count % process_every_n_frames == 0:
                # Process frame for face detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Detect faces with MTCNN
                    boxes, probs = mtcnn.detect(frame_rgb)
                    
                    if boxes is not None and len(boxes) > 0:
                        # Align faces using MTCNN extract to avoid extra detect calls
                        try:
                            faces_aligned = mtcnn.extract(Image.fromarray(frame_rgb), boxes, save_path=None)
                        except Exception:
                            faces_aligned = None

                        for i, box in enumerate(boxes):
                            if probs[i] is None or probs[i] <= 0.9:
                                continue
                            x1, y1, x2, y2 = [int(coord) for coord in box]
                            color = (0, 0, 255)
                            label = "Unknown"
                            status = "Not recognized"

                            # Recognition cooldown key by coarse position to reduce churn
                            face_pos_id = f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"
                            should_recognize = True
                            if face_pos_id in recognition_cooldown and recognition_cooldown[face_pos_id] > 0:
                                should_recognize = False
                                recognition_cooldown[face_pos_id] -= 1

                            if should_recognize and faces_aligned is not None and len(faces_aligned) > i:
                                try:
                                    face_tensor = faces_aligned[i]
                                    # Convert PIL image to normalized tensor if needed
                                    if not hasattr(face_tensor, 'ndim'):
                                        # Assume PIL.Image
                                        try:
                                            pil_img = face_tensor
                                            pil_img = pil_img.resize((160, 160))
                                            np_img = np.array(pil_img)
                                            if np_img.ndim == 3:
                                                np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
                                                face_tensor = torch.from_numpy(np_img).float() / 255.0
                                                face_tensor = (face_tensor - 0.5) / 0.5  # [-1, 1]
                                            else:
                                                raise ValueError('Unexpected aligned face array dims')
                                        except Exception as e:
                                            print(f"Error converting aligned face: {e}")
                                            face_tensor = None
                                    if face_tensor is not None and hasattr(face_tensor, 'ndim') and face_tensor.ndim == 3:
                                        with torch.no_grad():
                                            emb_tensor = facenet(face_tensor.unsqueeze(0).to(device))
                                        embedding = emb_tensor.detach().cpu().numpy()[0]
                                        # Recognize face
                                        roll_no, name, similarity = student_db.recognize_face(embedding)
                                        if roll_no:
                                            color = (0, 255, 0)
                                            label = f"{name} ({roll_no})"
                                            status = f"Similarity: {similarity:.2f}"
                                            already_marked = (roll_no in marked_attendance and attendance_mode in marked_attendance[roll_no])
                                            if not already_marked:
                                                current_time = datetime.now().strftime("%H:%M:%S")
                                                with open(attendance_file, 'a', newline='') as f:
                                                    writer = csv.writer(f)
                                                    writer.writerow([roll_no, name, current_time, attendance_mode])
                                                if roll_no not in marked_attendance:
                                                    marked_attendance[roll_no] = []
                                                marked_attendance[roll_no].append(attendance_mode)
                                                status = f"Marked: {attendance_mode}"
                                                attendance_marked = True
                                                play_success_sound()
                                    # Reset cooldown for this face position
                                    recognition_cooldown[face_pos_id] = cooldown_frames
                                except Exception as e:
                                    print(f"Error processing face: {e}")

                            # Update tracked faces with smoothing and TTL
                            best_id = None
                            best_iou = 0.0
                            for fid, entry in tracked_faces.items():
                                iou_val = compute_iou(entry['bbox'], [x1, y1, x2, y2])
                                if iou_val > best_iou:
                                    best_iou = iou_val
                                    best_id = fid
                            if best_iou > 0.3 and best_id is not None:
                                # Smooth update
                                px1, py1, px2, py2 = tracked_faces[best_id]['bbox']
                                sx1 = int(smooth_alpha * x1 + (1 - smooth_alpha) * px1)
                                sy1 = int(smooth_alpha * y1 + (1 - smooth_alpha) * py1)
                                sx2 = int(smooth_alpha * x2 + (1 - smooth_alpha) * px2)
                                sy2 = int(smooth_alpha * y2 + (1 - smooth_alpha) * py2)
                                tracked_faces[best_id].update({
                                    'bbox': [sx1, sy1, sx2, sy2],
                                    'color': color,
                                    'label': label,
                                    'status': status,
                                    'ttl': tracked_ttl_frames,
                                    'last_update': frame_count,
                                })
                            else:
                                new_id = f"f{len(tracked_faces)+1}_{frame_count}"
                                tracked_faces[new_id] = {
                                    'bbox': [x1, y1, x2, y2],
                                    'color': color,
                                    'label': label,
                                    'status': status,
                                    'ttl': tracked_ttl_frames,
                                    'last_update': frame_count,
                                }
                except Exception as e:
                    print(f"Error in face detection: {e}")

            # Draw tracked faces every frame to avoid flicker
            to_delete = []
            for fid, entry in tracked_faces.items():
                x1, y1, x2, y2 = entry['bbox']
                color = entry['color']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if entry.get('label'):
                    cv2.putText(frame, entry['label'], (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if entry.get('status'):
                    cv2.putText(frame, entry['status'], (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # Decrease TTL if not updated recently
                if frame_count - entry.get('last_update', 0) > 0:
                    entry['ttl'] -= 1
                if entry['ttl'] <= 0:
                    to_delete.append(fid)
            for fid in to_delete:
                tracked_faces.pop(fid, None)
            
            # Convert frame to bytes for streaming
            if CV2_AVAILABLE:
                # Use lower quality for better performance
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame = buffer.tobytes()
            else:
                # Without OpenCV, we already have the frame bytes from mock camera
                pass
            
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        except GeneratorExit:
            # Client disconnected; stop gracefully
            break
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            # Create an error frame to display the error message
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            if CV2_AVAILABLE:
                cv2.putText(error_frame, f"Error: {str(e)[:50]}...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(error_frame, "Attempting to recover...", (50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Convert error frame to bytes for streaming
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
            else:
                # Simple error visualization without OpenCV
                frame = create_simple_error_frame()
            
            # Yield error frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Try to recover by reinitializing the camera
            release_camera()
            time.sleep(2)  # Wait longer before trying to reconnect
            cap = get_camera()
            if cap is None:
                # If camera still can't be opened, break the loop
                break
            # Continue the loop to try again
            continue

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera, streaming_enabled
    # Ensure camera is initialized
    if camera is None:
        camera = get_camera()
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global streaming_enabled
    streaming_enabled = True
    # Ensure camera is ready
    if camera is None or (hasattr(camera, 'isOpened') and not camera.isOpened()):
        try:
            get_camera()
        except Exception:
            pass
    return jsonify({'status': 'ok', 'streaming': True})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global streaming_enabled
    streaming_enabled = False
    # Release device so the laptop camera light turns off
    release_camera()
    return jsonify({'status': 'ok', 'streaming': False})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global attendance_mode
    data = request.get_json()
    if data and 'mode' in data:
        attendance_mode = data['mode']
        return jsonify({'status': 'success', 'mode': attendance_mode})
    return jsonify({'status': 'error', 'message': 'Invalid request'})

@app.route('/check_attendance_marked')
def check_attendance_marked():
    # This endpoint will be polled by the client to check if attendance was marked
    global attendance_marked
    result = {'play_sound': attendance_marked}
    
    # Reset the flag after it's been checked
    if attendance_marked:
        attendance_marked = False
        
    return jsonify(result)

@app.route('/camera_status')
def camera_status():
    # Check if camera is available and working
    global camera
    
    status = {
        'available': False,
        'is_mock': False,
        'message': 'Camera not initialized'
    }
    
    if camera is not None:
        status['available'] = camera.isOpened()
        status['is_mock'] = isinstance(camera, MockCamera)
        
        if status['is_mock']:
            status['message'] = 'Using mock camera (no physical camera available)'
        elif status['available']:
            status['message'] = 'Camera is working properly'
        else:
            status['message'] = 'Camera is not working properly'
    
    return jsonify(status)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        roll_no = request.form.get('roll_no')
        name = request.form.get('name')
        
        if not roll_no or not name:
            flash('Roll number and name are required', 'error')
            return redirect(url_for('register'))
        
        # Store in session for the capture page
        return redirect(url_for('capture_face', roll_no=roll_no, name=name))
    
    return render_template('register.html')

@app.route('/capture_face')
def capture_face():
    roll_no = request.args.get('roll_no')
    name = request.args.get('name')
    
    if not roll_no or not name:
        flash('Roll number and name are required', 'error')
        return redirect(url_for('register'))
    
    return render_template('capture.html', roll_no=roll_no, name=name)

@app.route('/save_face', methods=['POST'])
def save_face():
    try:
        data = request.get_json()
        if not data or 'roll_no' not in data or 'name' not in data or (('image' not in data) and ('images' not in data)):
            return jsonify({'status': 'error', 'message': 'Invalid data'})
        
        roll_no = data['roll_no']
        name = data['name']
        images_data = []
        if 'images' in data and isinstance(data['images'], list) and len(data['images']) > 0:
            images_data = data['images']
        elif 'image' in data:
            images_data = [data['image']]
        
        # Check if face recognition is available
        if not FACE_RECOGNITION_AVAILABLE:
            # Even without face recognition, we can still register the student
            # but we'll just store their info without embeddings
            student_db.students[roll_no] = {
                'name': name,
                'embedding_path': ''
            }
            student_db.save_database()
            return jsonify({'status': 'success', 'message': f'Student {name} ({roll_no}) registered successfully!'})
        
        # Compute embedding(s) and average
        computed_embeddings = []
        if not CV2_AVAILABLE:
            # Without OpenCV, we can't process the image for face recognition
            # but we can still register the student
            student_db.students[roll_no] = {
                'name': name,
                'embedding_path': ''
            }
            student_db.save_database()
            return jsonify({'status': 'success', 'message': f'Student {name} ({roll_no}) registered successfully!'})
        if CV2_AVAILABLE:
            for image_data in images_data:
                try:
                    # Extract the base64 encoded image data
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(img_rgb)
                    # Use MTCNN to get aligned face tensor
                    face_tensor = mtcnn(face_pil)
                    if face_tensor is None:
                        continue
                    if hasattr(face_tensor, 'ndim') and face_tensor.ndim == 4:
                        face_tensor = face_tensor[0]
                    if not hasattr(face_tensor, 'ndim') or face_tensor.ndim != 3:
                        continue
                    with torch.no_grad():
                        emb_tensor = facenet(face_tensor.unsqueeze(0).to(device))
                    embedding = emb_tensor.detach().cpu().numpy()[0]
                    computed_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error computing embedding during registration: {e}")
                    continue
        
        if len(computed_embeddings) == 0:
            return jsonify({'status': 'error', 'message': 'No valid face detected in provided images'})
        
        # Average embeddings for robustness
        avg_embedding = np.mean(np.stack(computed_embeddings, axis=0), axis=0)
        
        # Save the student with averaged embedding
        student_db.add_student(roll_no, name, avg_embedding)
        return jsonify({'status': 'success', 'message': f'Student {name} ({roll_no}) registered successfully!'})
            
    except Exception as e:
        print(f"Error in save_face: {e}")
        return jsonify({'status': 'error', 'message': f'Error processing image: {str(e)}'})

@app.route('/attendance')
def attendance():
    date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    
    # Check for both old and new file formats
    attendance_files = [
        f'attendance/attendance_{date}.csv',  # Old format
        f'attendance/lab_{date}.csv',         # New format - lab
        f'attendance/class_{date}.csv'        # New format - class
    ]
    
    attendance_data = []
    for attendance_file in attendance_files:
        if os.path.exists(attendance_file):
            try:
                with open(attendance_file, 'r') as f:
                    reader = csv.reader(f)
                    headers = next(reader)  # Get headers
                    for row in reader:
                        if row:
                            # Ensure row has at least 4 elements
                            while len(row) < 4:
                                row.append('')
                            attendance_data.append(row)
            except Exception as e:
                print(f"Error reading attendance file {attendance_file}: {e}")
    
    return render_template('attendance.html', attendance_data=attendance_data, date=date)

@app.route('/download_attendance')
def download_attendance():
    date = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    mode = request.args.get('mode', 'all')  # 'all', 'lab', or 'class'
    
    if mode == 'all':
        # Combine all attendance files for the date
        attendance_files = [
            f'attendance/attendance_{date}.csv',  # Old format
            f'attendance/lab_{date}.csv',         # New format - lab
            f'attendance/class_{date}.csv'        # New format - class
        ]
        
        # Create a combined CSV
        combined_data = []
        headers = ['RollNo', 'Name', 'Time', 'Mode']
        
        for attendance_file in attendance_files:
            if os.path.exists(attendance_file):
                try:
                    with open(attendance_file, 'r') as f:
                        reader = csv.reader(f)
                        file_headers = next(reader)  # Skip headers
                        for row in reader:
                            if row:
                                # Ensure row has at least 4 elements
                                while len(row) < 4:
                                    row.append('unknown')
                                combined_data.append(row)
                except Exception as e:
                    print(f"Error reading file {attendance_file}: {e}")
        
        if not combined_data:
            flash(f'No attendance record found for {date}', 'info')
            return redirect(url_for('attendance'))
        
        # Create CSV response
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(combined_data)
        csv_data = output.getvalue()
        output.close()
        
        filename = f'attendance_{date}_all.csv'
    else:
        # Download specific mode
        attendance_file = f'attendance/{mode}_{date}.csv'
        
        if not os.path.exists(attendance_file):
            flash(f'No {mode} attendance record found for {date}', 'info')
            return redirect(url_for('attendance'))
        
        try:
            with open(attendance_file, 'r') as f:
                csv_data = f.read()
        except Exception as e:
            flash(f'Error reading attendance file: {e}', 'error')
            return redirect(url_for('attendance'))
        
        filename = f'attendance_{date}_{mode}.csv'
    
    response = Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )
    
    return response

@app.route('/students')
def students():
    student_list = []
    for roll_no, info in student_db.students.items():
        student_list.append({
            'roll_no': roll_no,
            'name': info['name'],
        })
    
    # Sort students by roll number
    student_list.sort(key=lambda x: x['roll_no'])
    
    return render_template('students.html', students=student_list)

@app.route('/delete_student/<roll_no>', methods=['POST'])
def delete_student(roll_no):
    if roll_no in student_db.students:
        try:
            # Get embedding path
            embedding_path = student_db.students[roll_no]['embedding_path']
            
            # Remove from database
            del student_db.students[roll_no]
            
            # Update embeddings array and names list
            if student_db.roll_numbers and roll_no in student_db.roll_numbers:
                indices_to_keep = [i for i, r in enumerate(student_db.roll_numbers) if r != roll_no]
                if indices_to_keep:
                    student_db.embeddings = student_db.embeddings[indices_to_keep]
                    student_db.names = [student_db.names[i] for i in indices_to_keep]
                    student_db.roll_numbers = [student_db.roll_numbers[i] for i in indices_to_keep]
                else:
                    student_db.embeddings = np.array([])
                    student_db.names = []
                    student_db.roll_numbers = []
            
            # Save updated database
            student_db.save_database()
            
            # Remove embedding file
            if os.path.exists(embedding_path):
                os.remove(embedding_path)
            
            flash(f'Student with roll number {roll_no} deleted successfully', 'success')
        except Exception as e:
            flash(f'Error deleting student: {e}', 'error')
    else:
        flash(f'Student with roll number {roll_no} not found', 'error')
    
    return redirect(url_for('students'))

@app.route('/statistics')
def statistics():
    """Display attendance statistics"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get attendance data for today
    lab_file = f'attendance/lab_{today}.csv'
    class_file = f'attendance/class_{today}.csv'
    
    stats = {
        'date': today,
        'total_students': len(student_db.students),
        'lab_attendance': 0,
        'class_attendance': 0,
        'lab_attendees': [],
        'class_attendees': []
    }
    
    # Count lab attendance
    if os.path.exists(lab_file):
        try:
            with open(lab_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and len(row) >= 2:
                        stats['lab_attendees'].append({'roll_no': row[0], 'name': row[1], 'time': row[2] if len(row) > 2 else ''})
            stats['lab_attendance'] = len(stats['lab_attendees'])
        except Exception as e:
            print(f"Error reading lab attendance: {e}")
    
    # Count class attendance
    if os.path.exists(class_file):
        try:
            with open(class_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and len(row) >= 2:
                        stats['class_attendees'].append({'roll_no': row[0], 'name': row[1], 'time': row[2] if len(row) > 2 else ''})
            stats['class_attendance'] = len(stats['class_attendees'])
        except Exception as e:
            print(f"Error reading class attendance: {e}")
    
    return render_template('statistics.html', stats=stats)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# Cleanup function
@app.teardown_appcontext
def cleanup(exception):
    # Removed automatic release on every request teardown to avoid
    # interrupting the live video stream. Cleanup is handled at process exit.
    pass

# Ensure camera is released when the process exits
atexit.register(release_camera)

# Additional utility functions
def get_attendance_summary(date=None):
    """Get attendance summary for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    lab_file = f'attendance/lab_{date}.csv'
    class_file = f'attendance/class_{date}.csv'
    
    summary = {
        'date': date,
        'lab': [],
        'class': [],
        'total_lab': 0,
        'total_class': 0
    }
    
    # Read lab attendance
    if os.path.exists(lab_file):
        try:
            with open(lab_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and len(row) >= 2:
                        summary['lab'].append({
                            'roll_no': row[0],
                            'name': row[1],
                            'time': row[2] if len(row) > 2 else ''
                        })
            summary['total_lab'] = len(summary['lab'])
        except Exception as e:
            print(f"Error reading lab attendance: {e}")
    
    # Read class attendance
    if os.path.exists(class_file):
        try:
            with open(class_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and len(row) >= 2:
                        summary['class'].append({
                            'roll_no': row[0],
                            'name': row[1],
                            'time': row[2] if len(row) > 2 else ''
                        })
            summary['total_class'] = len(summary['class'])
        except Exception as e:
            print(f"Error reading class attendance: {e}")
    
    return summary

@app.route('/api/attendance_summary')
def api_attendance_summary():
    """API endpoint to get attendance summary"""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    summary = get_attendance_summary(date)
    return jsonify(summary)

@app.route('/reset_attendance', methods=['POST'])
def reset_attendance():
    """Reset attendance for a specific date and mode"""
    data = request.get_json()
    
    if not data or 'date' not in data or 'mode' not in data:
        return jsonify({'status': 'error', 'message': 'Date and mode are required'})
    
    date = data['date']
    mode = data['mode']
    
    # Validate mode
    if mode not in ['lab', 'class']:
        return jsonify({'status': 'error', 'message': 'Invalid mode. Must be lab or class'})
    
    attendance_file = f'attendance/{mode}_{date}.csv'
    
    try:
        if os.path.exists(attendance_file):
            # Create backup before deletion
            backup_file = f'attendance/backup_{mode}_{date}_{int(time.time())}.csv'
            os.rename(attendance_file, backup_file)
            
            # Create new empty file with headers
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['RollNo', 'Name', 'Time', 'Mode'])
            
            return jsonify({
                'status': 'success', 
                'message': f'{mode.capitalize()} attendance for {date} has been reset. Backup saved as {backup_file}'
            })
        else:
            return jsonify({'status': 'error', 'message': f'No {mode} attendance file found for {date}'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error resetting attendance: {str(e)}'})

if __name__ == '__main__':
    try:
        ensure_directories()
        # Initialize camera at startup
        camera = get_camera()
        print("Starting Face Recognition Attendance System...")
        print(f"OpenCV Available: {CV2_AVAILABLE}")
        print(f"Face Recognition Available: {FACE_RECOGNITION_AVAILABLE}")
        print(f"Total Students in Database: {len(student_db.students)}")
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        release_camera()
    except Exception as e:
        print(f"Error starting application: {e}")
        release_camera()