import cv2
import numpy as np
import os

# Define the chair positions (these would be predefined based on the video or manually set)
chairs = {
    1: {"id": "person1", "coords": (0, 100, 150, 250)},  # Example: (x, y, width, height)
    2: {"id": "person2", "coords": (250, 100, 150, 250)},
    3: {"id": "person3", "coords": (600, 100, 150, 250)},
    4: {"id": "person4", "coords": (-10, -100, 150, 250)},
    # Add more chairs as needed
}

# Load Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known face images from folders
def load_known_faces():
    known_faces = {}
    
    # Load all images from person1 folder
    person1_folder = "extracted_faces/person1/"
    person2_folder = "extracted_faces/person2/"
    person3_folder = "extracted_faces/person3/"
    person4_folder = "extracted_faces/person4/"
    # Read all images from the "person1" folder
    person1_images = [cv2.imread(os.path.join(person1_folder, f)) for f in os.listdir(person1_folder) if f.endswith('.jpg')]
    person2_images = [cv2.imread(os.path.join(person2_folder, f)) for f in os.listdir(person2_folder) if f.endswith('.jpg')]
    person3_images = [cv2.imread(os.path.join(person3_folder, f)) for f in os.listdir(person3_folder) if f.endswith('.jpg')]
    person4_images = [cv2.imread(os.path.join(person4_folder, f)) for f in os.listdir(person4_folder) if f.endswith('.jpg')]
    # Store images in a dictionary, can be extended to include more persons
    known_faces["person1"] = person1_images
    known_faces["person2"] = person2_images
    known_faces["person3"] = person3_images
    known_faces["person4"] = person4_images
    return known_faces

# Check if the detected face matches any of the known faces using template matching
def find_person_in_frame(frame, known_faces):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    matched_faces = []
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        
        for person_id, known_face_list in known_faces.items():
            for known_face in known_face_list:
                resized_known_face = cv2.resize(known_face, (w, h))
                
                result = cv2.matchTemplate(face_image, resized_known_face, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.7:  # Adjust threshold if necessary
                    matched_faces.append((person_id, (x, y, w, h)))
    
    return matched_faces

# Main processing function
def check_persons_on_chairs(mismatch_folder="mismatched_frames"):
    # Create the folder if it doesn't exist
    if not os.path.exists(mismatch_folder):
        os.makedirs(mismatch_folder)
    
    # Load known faces from folders
    known_faces = load_known_faces()

    # Use the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return
    
    frame_count = 0  # To keep track of the frames
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons in the current frame
        matched_faces = find_person_in_frame(frame, known_faces)

        # Store the positions of persons detected
        person_positions = { "person1": None, "person2": None, "person3": None, "person4": None }
        
        # Loop through each person detected in the frame
        for person_id, (x, y, w, h) in matched_faces:
            # Draw a bounding box around the detected face (in green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Store the position of the detected person
            person_positions[person_id] = (x, y, w, h)
        
        # Check if person1 is sitting on chair2 and person2 on chair1
        mismatch_found = False
        for chair_id, chair_info in chairs.items():
            chair_x, chair_y, chair_w, chair_h = chair_info["coords"]
            assigned_person_id = chair_info["id"]
            
            if person_positions[assigned_person_id] is not None:
                x, y, w, h = person_positions[assigned_person_id]
                
                if chair_id == 1 and assigned_person_id == "person1" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                    print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                elif chair_id == 2 and assigned_person_id == "person2" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                    print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                elif chair_id == 3 and assigned_person_id == "person3" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                    print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                elif chair_id == 4 and assigned_person_id == "person4" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                    print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                else:
                    mismatch_found = True
                    # Save the image of the bounding box of the mismatched person
                    print(f"Mismatch detected: {assigned_person_id} is not sitting on chair {chair_id}")
                    bounding_box_image = frame[y:y + h, x:x + w]
                    mismatch_filename = f"{mismatch_folder}/frame_{frame_count}_person_{assigned_person_id}_chair_{chair_id}.jpg"
                    cv2.imwrite(mismatch_filename, bounding_box_image)
                    print(f"Image saved: {mismatch_filename}")
        
        # Display the frame
        cv2.imshow("Webcam Feed", frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

# Run the function with your webcam feed
check_persons_on_chairs()
