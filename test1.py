# import cv2
# import numpy as np

# Define the chair positions (these would be predefined based on the video or manually set)
# Each chair is represented by a bounding box (x, y, width, height) and the ID of the person assigned to that chair.
# chairs = {
#     1: {"id": "person1", "coords": (100, 200, 150, 100)},  # Example: (x, y, width, height)
#     2: {"id": "person2", "coords": (300, 200, 150, 100)},
#     Add more chairs as needed
# }

# Load Haar Cascade Classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known face images for matching (for simplicity, we just load person IDs and images)
# def load_known_faces():
#     known_faces = {}
    
#     Example of loading face images for each person (this would be done during registration)
#     Load images for person1 and person2, for instance
#     person1_image = cv2.imread("extracted_faces/person1.jpg")  # Replace with the correct file path
#     person2_image = cv2.imread("extracted_faces/person2.jpg")  # Replace with the correct file path
    
#     Resize or process the images if needed
#     known_faces["person1"] = person1_image
#     known_faces["person2"] = person2_image
    
#     return known_faces

# Check if the detected face matches any of the known faces (basic template matching)
# def find_person_in_frame(frame, known_faces):
#     Convert the frame to grayscale (Haar Cascade works in grayscale)
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     Detect faces using Haar Cascade
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     matched_faces = []
    
#     for (x, y, w, h) in faces:
#         Crop the face from the frame
#         face_image = frame[y:y+h, x:x+w]
        
#         Compare the detected face with known faces using basic template matching (for simplicity)
#         for person_id, known_face in known_faces.items():
#             Resize known faces to match the detected face size (simplified comparison)
#             resized_known_face = cv2.resize(known_face, (w, h))
            
#             Compare the detected face with the known face (this can be improved with better techniques)
#             result = cv2.matchTemplate(face_image, resized_known_face, cv2.TM_CCOEFF_NORMED)
#             _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
#             If a match is found with a threshold, consider it a valid match
#             if max_val > 0.8:  # You can adjust the threshold value
#                 matched_faces.append((person_id, (x, y, w, h)))
    
#     return matched_faces

# Main processing function
# def check_persons_on_chairs(video_path):
#     Load known faces
#     known_faces = load_known_faces()

#     Open video file
#     cap = cv2.VideoCapture(video_path)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         Detect persons in the current frame
#         matched_faces = find_person_in_frame(frame, known_faces)

#         Loop through each person detected in the frame
#         for person_id, (x, y, w, h) in matched_faces:
#             Check if the detected face is in the assigned chair region
#             for chair_id, chair_info in chairs.items():
#                 chair_x, chair_y, chair_w, chair_h = chair_info["coords"]
                
#                 Check if the face bounding box intersects with the chair's coordinates
#                 if chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h:
#                     print(f"{person_id} is sitting on chair {chair_id}")
#                 else:
#                     print(f"{person_id} is NOT sitting on chair {chair_id}")
        
#         Display the frame (optional)
#         cv2.imshow("Video", frame)
        
#         Break if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# Run the function with your video
# video_path = "sit.mp4"  # Path to your uploaded video
# check_persons_on_chairs(video_path)
# In this script, we define the positions of chairs and the persons assigned to those chairs. We load known face images
# import cv2
# import numpy as np

# # Define the chair positions (these would be predefined based on the video or manually set)
# # Each chair is represented by a bounding box (x, y, width, height) and the ID of the person assigned to that chair.
# chairs = {
#     1: {"id": "person1", "coords": (0, 100, 150, 250)},  # Example: (x, y, width, height)
#     2: {"id": "person2", "coords": (250, 100, 150, 250)},
#     # Add more chairs as needed
# }

# # Load Haar Cascade Classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load known face images for matching (for simplicity, we just load person IDs and images)
# def load_known_faces():
#     known_faces = {}
    
#     # Example of loading face images for each person (this would be done during registration)
#     # Load images for person1 and person2, for instance
#     person1_image = cv2.imread("extracted_faces/person1.jpg")  # Replace with the correct file path
#     person2_image = cv2.imread("extracted_faces/person2.jpg")  # Replace with the correct file path
    
#     # Resize or process the images if needed
#     known_faces["person1"] = person1_image
#     known_faces["person2"] = person2_image
    
#     return known_faces

# # Check if the detected face matches any of the known faces (basic template matching)
# def find_person_in_frame(frame, known_faces):
#     # Convert the frame to grayscale (Haar Cascade works in grayscale)
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces using Haar Cascade
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     matched_faces = []
    
#     for (x, y, w, h) in faces:
#         # Crop the face from the frame
#         face_image = frame[y:y+h, x:x+w]
        
#         # Compare the detected face with known faces using basic template matching (for simplicity)
#         for person_id, known_face in known_faces.items():
#             # Resize known faces to match the detected face size (simplified comparison)
#             resized_known_face = cv2.resize(known_face, (w, h))
            
#             # Compare the detected face with the known face (this can be improved with better techniques)
#             result = cv2.matchTemplate(face_image, resized_known_face, cv2.TM_CCOEFF_NORMED)
#             _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
#             # If a match is found with a threshold, consider it a valid match
#             if max_val > 0.7:  # You can adjust the threshold value
#                 matched_faces.append((person_id, (x, y, w, h)))
    
#     return matched_faces

# # Main processing function
# def check_persons_on_chairs(video_path):
#     # Load known faces
#     known_faces = load_known_faces()

#     # Open video file
#     cap = cv2.VideoCapture(video_path)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Detect persons in the current frame
#         matched_faces = find_person_in_frame(frame, known_faces)

#         # Loop through each person detected in the frame
#         for person_id, (x, y, w, h) in matched_faces:
#             # Draw a bounding box around the detected face (in green)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
#             # Check if the detected face is in the assigned chair region
#             for chair_id, chair_info in chairs.items():
#                 chair_x, chair_y, chair_w, chair_h = chair_info["coords"]
                
#                 # Draw the chair's bounding box (in blue)
#                 cv2.rectangle(frame, (chair_x, chair_y), (chair_x + chair_w, chair_y + chair_h), (255, 0, 0), 2)
                
#                 # Check if the face's bounding box is inside the chair's bounding box
#                 if chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h:
#                     print(f"{person_id} is sitting on chair {chair_id}")
#                 else:
#                     print(f"{person_id} is NOT sitting on chair {chair_id}")
        
#         # Display the frame (optional)
#         cv2.imshow("Video", frame)
        
#         # Break if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# # Run the function with your video
# video_path = "sit.mp4"  # Path to your uploaded video
# check_persons_on_chairs(video_path)
# import cv2
# import numpy as np
# import os

# # Define the chair positions (these would be predefined based on the video or manually set)
# chairs = {
#     1: {"id": "person1", "coords": (0, 100, 150, 250)},  # Example: (x, y, width, height)
#     2: {"id": "person2", "coords": (250, 100, 150, 250)},
#     # Add more chairs as needed
# }

# # Load Haar Cascade Classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load known face images for matching (for simplicity, we just load person IDs and images)
# def load_known_faces():
#     known_faces = {}
    
#     # Example of loading face images for each person (this would be done during registration)
#     # Load images for person1 and person2, for instance
#     person1_image = cv2.imread("extracted_faces/person2")  # Replace with the correct file path
#     person2_image = cv2.imread("extracted_faces/person1")  # Replace with the correct file path
    
#     # Resize or process the images if needed
#     known_faces["person1"] = person1_image
#     known_faces["person2"] = person2_image
    
#     return known_faces

# # Check if the detected face matches any of the known faces (basic template matching)
# def find_person_in_frame(frame, known_faces):
#     # Convert the frame to grayscale (Haar Cascade works in grayscale)
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces using Haar Cascade
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     matched_faces = []
    
#     for (x, y, w, h) in faces:
#         # Crop the face from the frame
#         face_image = frame[y:y+h, x:x+w]
        
#         # Compare the detected face with known faces using basic template matching (for simplicity)
#         for person_id, known_face in known_faces.items():
#             # Resize known faces to match the detected face size (simplified comparison)
#             resized_known_face = cv2.resize(known_face, (w, h))
            
#             # Compare the detected face with the known face (this can be improved with better techniques)
#             result = cv2.matchTemplate(face_image, resized_known_face, cv2.TM_CCOEFF_NORMED)
#             _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
#             # If a match is found with a threshold, consider it a valid match
#             if max_val > 0.7:  # You can adjust the threshold value
#                 matched_faces.append((person_id, (x, y, w, h)))
    
#     return matched_faces

# # Main processing function
# def check_persons_on_chairs(video_path, mismatch_folder="mismatched_frames"):
#     # Create the folder if it doesn't exist
#     if not os.path.exists(mismatch_folder):
#         os.makedirs(mismatch_folder)
    
#     # Load known faces
#     known_faces = load_known_faces()

#     # Open video file
#     cap = cv2.VideoCapture(video_path)
    
#     frame_count = 0  # To keep track of the frames
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Detect persons in the current frame
#         matched_faces = find_person_in_frame(frame, known_faces)

#         # Store the positions of persons detected
#         person_positions = { "person1": None, "person2": None }
        
#         # Loop through each person detected in the frame
#         for person_id, (x, y, w, h) in matched_faces:
#             # Draw a bounding box around the detected face (in green)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
#             # Store the position of the detected person
#             person_positions[person_id] = (x, y, w, h)
        
#         # Check if person1 is sitting on chair2 and person2 on chair1
#         mismatch_found = False
#         for chair_id, chair_info in chairs.items():
#             chair_x, chair_y, chair_w, chair_h = chair_info["coords"]
#             assigned_person_id = chair_info["id"]
            
#             if person_positions[assigned_person_id] is not None:
#                 x, y, w, h = person_positions[assigned_person_id]
                
#                 # Check if the person is sitting in the assigned chair
#                 if chair_id == 1 and assigned_person_id == "person1" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
#                     print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
#                 elif chair_id == 2 and assigned_person_id == "person2" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
#                     print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
#                 else:
#                     mismatch_found = True
#                     # Save the image of the bounding box of the mismatched person
#                     print(f"Mismatch detected: {assigned_person_id} is not sitting on chair {chair_id}")
#                     bounding_box_image = frame[y:y + h, x:x + w]
#                     mismatch_filename = f"{mismatch_folder}/frame_{frame_count}_person_{assigned_person_id}_chair_{chair_id}.jpg"
#                     cv2.imwrite(mismatch_filename, bounding_box_image)
#                     print(f"Image saved: {mismatch_filename}")
        
#         # Display the frame (optional)
#         cv2.imshow("Video", frame)
        
#         # Break if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#         frame_count += 1
    
#     cap.release()
#     cv2.destroyAllWindows()

# # Run the function with your video
# video_path = "sit.mp4"  # Path to your uploaded video
# check_persons_on_chairs(video_path)
import cv2
import numpy as np
import os

# Define the chair positions (these would be predefined based on the video or manually set)
chairs = {
    1: {"id": "person4", "coords": (0, 100, 150, 250)},  # Example: (x, y, width, height)
    2: {"id": "person2", "coords": (250, 100, 150, 250)},
    3: {"id": "person3", "coords": (600, 100, 150, 250)},
    4: {"id": "person1", "coords": (-10, -100, 150, 250)},
    #5: {"id": "person4", "coords": (-250, -100, 150, 250)},
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
    #person5_folder = "extracted_faces/person5/"
    # Read all images from the "person1" folder
    person1_images = [cv2.imread(os.path.join(person1_folder, f)) for f in os.listdir(person1_folder) if f.endswith('.jpg')]
    person2_images = [cv2.imread(os.path.join(person2_folder, f)) for f in os.listdir(person2_folder) if f.endswith('.jpg')]
    person3_images = [cv2.imread(os.path.join(person3_folder, f)) for f in os.listdir(person3_folder) if f.endswith('.jpg')]
    person4_images = [cv2.imread(os.path.join(person4_folder, f)) for f in os.listdir(person4_folder) if f.endswith('.jpg')]
   # person5_images = [cv2.imread(os.path.join(person5_folder, f)) for f in os.listdir(person5_folder) if f.endswith('.jpg')]
    # Store images in a dictionary, can be extended to include more persons
    known_faces["person1"] = person1_images
    known_faces["person2"] = person2_images
    known_faces["person3"] = person3_images
    known_faces["person4"] = person4_images
    #known_faces["person5"] = person5_images 
    return known_faces

# Check if the detected face matches any of the known faces using template matching
def find_person_in_frame(frame, known_faces):
    # Convert the frame to grayscale (Haar Cascade works in grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    matched_faces = []
    
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face_image = frame[y:y+h, x:x+w]
        
        # Compare the detected face with known faces using basic template matching (for simplicity)
        for person_id, known_face_list in known_faces.items():
            for known_face in known_face_list:
                # Resize known faces to match the detected face size (simplified comparison)
                resized_known_face = cv2.resize(known_face, (w, h))
                
                # Compare the detected face with the known face using template matching
                result = cv2.matchTemplate(face_image, resized_known_face, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                # If a match is found with a threshold, consider it a valid match
                if max_val > 0.7:  # Adjust threshold if necessary
                    matched_faces.append((person_id, (x, y, w, h)))
    
    return matched_faces

# Main processing function
def check_persons_on_chairs(video_path, mismatch_folder="mismatched_frames"):
    # Create the folder if it doesn't exist
    if not os.path.exists(mismatch_folder):
        os.makedirs(mismatch_folder)
    
    # Load known faces from folders
    known_faces = load_known_faces()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0  # To keep track of the frames
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons in the current frame
        matched_faces = find_person_in_frame(frame, known_faces)

        # Store the positions of persons detected
        person_positions = { "person1": None, "person2": None,"person3": None,"person4": None }
        
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
                
                # Check if the person is sitting in the assigned chair
                if chair_id == 1 and assigned_person_id == "person1" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                    print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                elif chair_id == 2 and assigned_person_id == "person2" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                    print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                elif chair_id == 3 and assigned_person_id == "person3" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                    print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                elif chair_id == 4 and assigned_person_id == "person4" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                    print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                #elif chair_id == 5 and assigned_person_id == "person5" and (chair_x < x + w and x < chair_x + chair_w and chair_y < y + h and y < chair_y + chair_h):
                   # print(f"{assigned_person_id} is sitting correctly on chair {chair_id}")
                else:
                    mismatch_found = True
                    # Save the image of the bounding box of the mismatched person
                    print(f"Mismatch detected: {assigned_person_id} is not sitting on chair {chair_id}")
                    bounding_box_image = frame[y:y + h, x:x + w]
                    mismatch_filename = f"{mismatch_folder}/frame_{frame_count}_person_{assigned_person_id}_chair_{chair_id}.jpg"
                    cv2.imwrite(mismatch_filename, bounding_box_image)
                    print(f"Image saved: {mismatch_filename}")
        
        # Display the frame (optional)
        cv2.imshow("Video", frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

# Run the function with your video
video_path ="sit.mp4" # Path to your uploaded video
check_persons_on_chairs(video_path)





