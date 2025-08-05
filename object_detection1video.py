import cv2
import torch

# Load the pre-trained YOLO-NAS model (replace with correct model path if necessary)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Initialize the video stream (webcam or video file)
cap = cv2.VideoCapture(0)  # 0 is the default camera, change to the path of the video file if needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)
    results.render()
    output_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    cv2.imshow("Detection", output_frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
