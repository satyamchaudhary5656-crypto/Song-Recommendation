import cv2
from fer import FER

# mtcnn=True is accurate but can be slow. Set to False for speed.
detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect_emotions(frame)
    
    if results:
        for face in results:
            # Get the box coordinates
            (x, y, w, h) = face["box"]
            
            # FIX: Extract emotion from the specific 'face' result 
            # instead of re-scanning the whole frame
            emotions = face["emotions"]
            
            # Find the emotion with the highest score manually
            top_emotion = max(emotions, key=emotions.get)
            score = emotions[top_emotion]

            # Draw the rectangle and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{top_emotion} ({score:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()