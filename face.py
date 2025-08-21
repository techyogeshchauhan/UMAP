# Try to import OpenCV, use mock if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("WARNING: OpenCV (cv2) is not installed. Using mock camera.")
    CV2_AVAILABLE = False

# Try to import numpy, use mock if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("WARNING: NumPy is not installed. Using mock implementation.")
    NUMPY_AVAILABLE = False
    # Create a simple mock for numpy arrays
    class MockArray:
        def __init__(self, *args, **kwargs):
            self.shape = (0,)
            self.size = 0
        def flatten(self):
            return self
        def reshape(self, *args):
            return self
    
    # Create a minimal numpy mock module
    class MockNumPy:
        def __init__(self):
            pass
        def array(self, *args, **kwargs):
            return MockArray()
        def zeros(self, *args, **kwargs):
            return MockArray()
        def ones(self, *args, **kwargs):
            return MockArray()
        def vstack(self, *args, **kwargs):
            return MockArray()
        def mean(self, *args, **kwargs):
            return MockArray()
        def sqrt(self, *args):
            return 0
        def sum(self, *args, **kwargs):
            return 0
        def argmin(self, *args):
            return 0
        def save(self, *args, **kwargs):
            pass
        def load(self, *args, **kwargs):
            return MockArray()
    
    # Create mock numpy module
    np = MockNumPy()

import os
import csv
import datetime
from datetime import datetime
import time

# Try to import face recognition libraries
try:
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from facenet_pytorch import InceptionResnetV1, MTCNN
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

# Initialize face recognition components only if available
if FACE_RECOGNITION_AVAILABLE:
    # Check for GPU availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize MTCNN for face detection with optimized parameters for speed
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=40,  # Increased min_face_size for faster detection
        thresholds=[0.7, 0.8, 0.8],  # Higher thresholds for faster but still accurate detection
        factor=0.8,  # Larger step size in the image pyramid for faster processing
        post_process=True,
        keep_all=True,  # Keep all faces for better tracking
        device=device,
        select_largest=False  # Don't just select largest face
    )

    # Initialize FaceNet for face recognition and ensure it's in evaluation mode
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Apply model optimization if using CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Use cudnn auto-tuner to find the best algorithm
else:
    # Create dummy device and models when face recognition is not available
    device = None
    mtcnn = MTCNN()
    resnet = InceptionResnetV1()

# Create directories if they don't exist
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('attendance'):
    os.makedirs('attendance')
if not os.path.exists('faces'):
    os.makedirs('faces')

# Load or create student database
class StudentDatabase:
    def __init__(self):
        self.students = {}
        self.embeddings = np.array([])
        self.names = []
        self.roll_numbers = []
        self.load_database()
        
    def load_database(self):
        if os.path.exists('data/students.csv'):
            with open('data/students.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        roll_no, name, embedding_path = row
                        self.students[roll_no] = {
                            'name': name,
                            'embedding_path': embedding_path
                        }
                        if os.path.exists(embedding_path):
                            embedding = np.load(embedding_path)
                            if self.embeddings.size == 0:
                                self.embeddings = embedding.reshape(1, -1)
                            else:
                                self.embeddings = np.vstack([self.embeddings, embedding])
                            self.names.append(name)
                            self.roll_numbers.append(roll_no)
        
    def add_student(self, roll_no, name, embedding):
        # Ensure embedding is 1D array
        embedding = embedding.flatten()
        embedding_path = f'faces/{roll_no}_{name.replace(" ", "_")}.npy'
        np.save(embedding_path, embedding)
        
        self.students[roll_no] = {
            'name': name,
            'embedding_path': embedding_path
        }
        
        if self.embeddings.size == 0:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.names.append(name)
        self.roll_numbers.append(roll_no)
        
        self.save_database()
    
    def save_database(self):
        with open('data/students.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['RollNo', 'Name', 'EmbeddingPath'])
            for roll_no, info in self.students.items():
                writer.writerow([roll_no, info['name'], info['embedding_path']])
    
    def recognize_face(self, embedding, threshold=0.8):
        if not self.embeddings.size:
            return None, None, None
        
        # Ensure input embedding is 1D
        embedding = embedding.flatten()
        
        # Calculate distances using optimized vectorized operations
        # This is faster than element-wise subtraction and norm calculation
        distances = np.sqrt(np.sum((self.embeddings - embedding)**2, axis=1))
        
        # Get minimum distance and index
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        
        # Use a slightly higher threshold for better recognition
        if min_distance < threshold:
            return self.roll_numbers[min_index], self.names[min_index], min_distance
        else:
            return None, None, None

# Initialize student database
student_db = StudentDatabase()

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
                self.width = value
            elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = value
        else:
            # Handle property IDs without OpenCV
            if prop_id == 3:  # CAP_PROP_FRAME_WIDTH
                self.width = value
            elif prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
                self.height = value
        return True

def get_mock_camera():
    print("Initializing mock camera")
    return MockCamera()

# Camera handling with fallback mechanism
def get_camera():
    # If OpenCV is not available, always use mock camera
    if not CV2_AVAILABLE:
        print("OpenCV not available. Using mock camera.")
        return get_mock_camera()
        
    try:
        # Try to open the camera with multiple attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                # Set camera properties for better performance
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Reduce resolution
                print(f"Camera opened successfully on attempt {attempt+1}")
                return camera
            else:
                print(f"Attempt {attempt+1}/{max_attempts}: Could not open camera.")
                if camera is not None:
                    camera.release()
                time.sleep(1)  # Wait before retrying
        
        print("Error: Failed to open camera after multiple attempts. Using mock camera.")
    except Exception as e:
        print(f"Error initializing camera: {e}. Using mock camera.")
    
    # Return a mock camera as fallback
    return get_mock_camera()

# Data collection function
def collect_data():
    cap = get_camera()
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Data Collection Mode")
    print("Press 'c' to capture, 'q' to quit")
    
    roll_no = input("Enter student roll number: ")
    name = input("Enter student name: ")
    
    if roll_no in student_db.students:
        print(f"Student with roll number {roll_no} already exists.")
        response = input("Do you want to update? (y/n): ")
        if response.lower() != 'y':
            cap.release()
            return
    
    count = 0
    embeddings = []
    
    while count < 10:  # Capture 10 samples
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect face if OpenCV and face recognition are available
        if CV2_AVAILABLE and FACE_RECOGNITION_AVAILABLE:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Display message if face recognition is not available
            if CV2_AVAILABLE:
                cv2.putText(frame, "Face recognition not available", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Cannot detect faces", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            boxes = None
        
        if CV2_AVAILABLE:
            cv2.putText(frame, f"Captures: {count}/10", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
        else:
            print(f"Captures: {count}/10")
        
        if CV2_AVAILABLE:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                capture_now = True
            else:
                capture_now = False
        else:
            # Without OpenCV, use input for capturing
            print("Press 'c' to capture, 'q' to quit")
            user_input = input().lower()
            if user_input == 'q':
                break
            elif user_input == 'c':
                capture_now = True
            else:
                capture_now = False
                
        if capture_now:
            if boxes is not None and len(boxes) > 0:
                try:
                    # Extract and align face
                    face = mtcnn(frame_rgb)
                    if face is not None:
                        # Get embedding
                        embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                        embeddings.append(embedding)
                        count += 1
                        print(f"Captured sample {count}")
                    else:
                        print("No face detected. Try again.")
                except Exception as e:
                    print(f"Error capturing face: {e}")
            elif not FACE_RECOGNITION_AVAILABLE:
                # Create dummy data for testing when face recognition isn't available
                dummy_embedding = np.zeros((1, 512))  # Assuming 512-dimensional embeddings
                embeddings.append(dummy_embedding)
                count += 1
                print(f"Created dummy data {count}/10 for testing (face recognition unavailable)")
            else:
                print("No face detected. Try again.")
    
    if embeddings:
        # Average embeddings for better accuracy
        avg_embedding = np.mean(embeddings, axis=0)
        student_db.add_student(roll_no, name, avg_embedding)
        print(f"Student {name} ({roll_no}) added successfully!")
    else:
        print("No valid faces captured.")
    
    cap.release()
    cv2.destroyAllWindows()

# Attendance system
def mark_attendance():
    cap = get_camera()
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Load attendance file or create new
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = f'attendance/attendance_{today}.csv'
    
    # Initialize attendance record if not exists
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['RollNo', 'Name', 'Time'])
    
    # Load already marked attendance
    marked_attendance = set()
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    marked_attendance.add(row[0])  # Roll number
    
    print("Attendance Mode - Press 'q' to quit")
    
    # Check if face recognition is available
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition not available. Running in limited mode.")
        print("You will need to manually enter student information.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            if CV2_AVAILABLE:
                # Display status
                status = f"Marked: {len(marked_attendance)} students"
                cv2.putText(frame, "FACE RECOGNITION UNAVAILABLE", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'm' to mark attendance manually", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Attendance System', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    # Manual attendance entry
                    roll_no = input("Enter student roll number: ")
                    if roll_no in student_db.students:
                        name = student_db.students[roll_no]['name']
                        if roll_no not in marked_attendance:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            with open(attendance_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([roll_no, name, current_time])
                            marked_attendance.add(roll_no)
                            print(f"Attendance marked for {name} ({roll_no}) at {current_time}")
                        else:
                            print(f"Attendance already marked for {name} ({roll_no})")
                    else:
                        print(f"Student with roll number {roll_no} not found in database.")
            else:
                # Without OpenCV, use text-based interface
                print("\nOptions: 'm' to mark attendance manually, 'q' to quit")
                user_input = input().lower()
                if user_input == 'q':
                    break
                elif user_input == 'm':
                    roll_no = input("Enter student roll number: ")
                    if roll_no in student_db.students:
                        name = student_db.students[roll_no]['name']
                        if roll_no not in marked_attendance:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            with open(attendance_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([roll_no, name, current_time])
                            marked_attendance.add(roll_no)
                            print(f"Attendance marked for {name} ({roll_no}) at {current_time}")
                        else:
                            print(f"Attendance already marked for {name} ({roll_no})")
                    else:
                        print(f"Student with roll number {roll_no} not found in database.")
    else:
        # Full face recognition mode
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)
            
            recognized_rolls = []
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    # Extract face
                    face_img = frame_rgb[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    
                    try:
                        # Get face embedding
                        face = mtcnn(Image.fromarray(face_img))
                        if face is None:
                            continue
                        
                        embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                        
                        # Recognize face
                        roll_no, name, distance = student_db.recognize_face(embedding)
                        
                        if roll_no:
                            color = (0, 255, 0)  # Green for recognized
                            label = f"{name} ({roll_no})"
                            recognized_rolls.append(roll_no)
                            
                            # Mark attendance if not already marked
                            if roll_no not in marked_attendance:
                                current_time = datetime.now().strftime("%H:%M:%S")
                                with open(attendance_file, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([roll_no, name, current_time])
                                marked_attendance.add(roll_no)
                                print(f"Attendance marked for {name} ({roll_no}) at {current_time}")
                        else:
                            color = (0, 0, 255)  # Red for unrecognized
                            label = "Unknown"
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except Exception as e:
                        print(f"Error processing face: {e}")
            
            # Display status
            status = f"Marked: {len(marked_attendance)} students"
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Attendance System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if CV2_AVAILABLE:
        cv2.destroyAllWindows()
    print(f"Attendance saved to {attendance_file}")

# View attendance records
def view_attendance():
    date = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    attendance_file = f'attendance/attendance_{date}.csv'
    
    if not os.path.exists(attendance_file):
        print(f"No attendance record found for {date}")
        return
    
    print(f"\nAttendance for {date}:")
    print("-" * 40)
    
    attendance_data = []
    with open(attendance_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                print(f"{row[0]}: {row[1]} - {row[2]}")
                attendance_data.append((row[0], row[1], row[2]))
    
    print("-" * 40)
    
    # If OpenCV is available, show a visual representation
    if CV2_AVAILABLE:
        try:
            # Create a blank image
            height = 50 + (len(attendance_data) * 30) + 50  # Header + rows + footer
            width = 800
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Add header
            cv2.putText(img, f"Attendance for {date}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.line(img, (20, 40), (width-20, 40), (0, 0, 0), 1)
            
            # Add column headers
            cv2.putText(img, "Roll No", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(img, "Name", (150, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(img, "Time", (500, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            cv2.line(img, (20, 80), (width-20, 80), (0, 0, 0), 1)
            
            # Add data rows
            for i, (roll_no, name, time) in enumerate(attendance_data):
                y = 110 + (i * 30)
                cv2.putText(img, roll_no, (20, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(img, name, (150, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(img, time, (500, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Show the image
            cv2.imshow("Attendance Report", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying visual attendance: {e}")

# Export attendance to Excel format
def export_attendance():
    date = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    attendance_file = f'attendance/attendance_{date}.csv'
    export_file = f'attendance/attendance_{date}_export.csv'
    
    if not os.path.exists(attendance_file):
        print(f"No attendance record found for {date}")
        return
    
    # Read and reformat attendance data
    attendance_data = {}
    with open(attendance_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row and len(row) >= 3:
                roll_no, name, time = row
                attendance_data[roll_no] = {'name': name, 'time': time}
    
    # Try to use pandas for Excel export if available
    try:
        import pandas as pd
        # Convert to DataFrame
        df_data = []
        for roll_no in sorted(attendance_data.keys()):
            info = attendance_data[roll_no]
            df_data.append([roll_no, info['name'], info['time'], 'Present'])
            
        df = pd.DataFrame(df_data, columns=['Roll Number', 'Name', 'Time', 'Status'])
        excel_file = f'attendance/attendance_{date}_export.xlsx'
        df.to_excel(excel_file, index=False)
        print(f"Attendance exported to Excel: {excel_file}")
    except ImportError:
        # Fallback to CSV if pandas is not available
        # Write to export file with additional formatting
        with open(export_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Roll Number', 'Name', 'Time', 'Status'])
            for roll_no in sorted(attendance_data.keys()):
                info = attendance_data[roll_no]
                writer.writerow([roll_no, info['name'], info['time'], 'Present'])
        
        print(f"Pandas not available. Attendance exported to CSV: {export_file}")

# Main menu
def main():
    # Check for required libraries and display status
    print("\nFace Recognition Attendance System")
    print("-" * 40)
    
    if not CV2_AVAILABLE:
        print("WARNING: OpenCV is not available. Some features will be limited.")
    if not FACE_RECOGNITION_AVAILABLE:
        print("WARNING: Face recognition libraries are not available. Face detection will not work.")
    
    print("-" * 40)
    
    while True:
        print("\nFace Recognition Attendance System")
        print("1. Add Student")
        print("2. Mark Attendance")
        print("3. View Attendance")
        print("4. Export Attendance")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            if not CV2_AVAILABLE or not FACE_RECOGNITION_AVAILABLE:
                print("WARNING: Face recognition or OpenCV is not available.")
                print("Registration will create dummy data for testing purposes.")
                proceed = input("Do you want to proceed? (y/n): ").lower()
                if proceed != 'y':
                    continue
            collect_data()
        elif choice == '2':
            if not CV2_AVAILABLE or not FACE_RECOGNITION_AVAILABLE:
                print("WARNING: Face recognition or OpenCV is not available.")
                print("Attendance will be marked manually without face recognition.")
                proceed = input("Do you want to proceed? (y/n): ").lower()
                if proceed != 'y':
                    continue
            mark_attendance()
        elif choice == '3':
            view_attendance()
        elif choice == '4':
            export_attendance()
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()