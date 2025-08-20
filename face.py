import cv2
import numpy as np
import os
import csv
import datetime
from datetime import datetime
import time
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

# Check for GPU availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize MTCNN for face detection
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Initialize FaceNet for face recognition
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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
        
        # Calculate distances
        distances = np.linalg.norm(self.embeddings - embedding, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        
        if min_distance < threshold:
            return self.roll_numbers[min_index], self.names[min_index], min_distance
        else:
            return None, None, None

# Initialize student database
student_db = StudentDatabase()

# Data collection function
def collect_data():
    cap = cv2.VideoCapture(0)
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
        
        # Detect face
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Captures: {count}/10", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
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
    cap = cv2.VideoCapture(0)
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
    
    with open(attendance_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                print(f"{row[0]}: {row[1]} - {row[2]}")
    
    print("-" * 40)

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
    
    # Write to export file with additional formatting
    with open(export_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Roll Number', 'Name', 'Time', 'Status'])
        for roll_no in sorted(attendance_data.keys()):
            info = attendance_data[roll_no]
            writer.writerow([roll_no, info['name'], info['time'], 'Present'])
    
    print(f"Attendance exported to {export_file}")

# Main menu
def main():
    while True:
        print("\nFace Recognition Attendance System")
        print("1. Add Student")
        print("2. Mark Attendance")
        print("3. View Attendance")
        print("4. Export Attendance")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            collect_data()
        elif choice == '2':
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