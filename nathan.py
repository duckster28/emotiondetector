import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze emotions
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Check if faces are detected
        if isinstance(analysis, list) and len(analysis) > 0:
            for face in analysis:
                if 'region' in face and isinstance(face['region'], dict):
                    region = face['region']
                    
                    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)


                    emotion = face.get('dominant_emotion', 'Unknown')

                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if (emotion == 'happy'):
                        cv2.putText(frame, "geeked", (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    elif (emotion == 'sad'):
                        cv2.putText(frame, "locked in", (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    except Exception as e:
        print("Error:", e)

    # Display the frame
    cv2.imshow("Facial Emotion Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
