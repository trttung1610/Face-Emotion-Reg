import os
import tempfile
import torch
import cv2
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

def process_emotion(input_video):
    try:
        # Load the pre-trained CNN model
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 7)  # 7 classes for emotions

        # Load the pre-trained weights
        model.eval()

        # Define emotions
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        # Initialize MTCNN for face detection
        mtcnn = MTCNN()

        # Open the video capture
        cap = cv2.VideoCapture(input_video.name)

        if not cap.isOpened():
            return "Error: Could not open video capture."

        # Create a temporary directory to store the frames
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        # Define image transformations
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Process each frame of the video
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if frame is None:
                print("Error: Empty frame.")
                continue

            frame_count += 1

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_boxes, _ = mtcnn.detect(pil_frame)

            if face_boxes is None:
                print(f"No face detected in frame {frame_count}.")
            else:
                for box in face_boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    w = x2 - x1
                    h = y2 - y1

                    face_roi = frame[y1:y2, x1:x2]
                    pil_face = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

                    input_tensor = preprocess(pil_face)
                    input_batch = input_tensor.unsqueeze(0)

                    with torch.no_grad():
                        output = model(input_batch)

                    predicted_emotion = output.argmax().item()

                    emotion_label = emotions[predicted_emotion]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, emotion_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    frame_path = os.path.join(temp_dir, f"{frame_count:04d}.png")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)

        cap.release()
        cv2.destroyAllWindows()

        if not frame_paths:
            return "No faces detected in the video."

        # Convert the frames to a video
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Get dimensions from the first frame
        sample_frame = cv2.imread(frame_paths[0])
        if sample_frame is None:
            return "Error: Unable to read sample frame."

        frame_height, frame_width, _ = sample_frame.shape

        out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
                os.remove(frame_path)
            else:
                print(f"Warning: Unable to read frame {frame_path}")

        out.release()

        return output_path

    except Exception as e:
        return f"An error occurred: {str(e)}"
