import cv2
import os
import time

# Parameters
name = input("Enter file's name: ")
output_folder = f"images/data/{name}/"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
capturing = False
img_count = len(os.listdir(output_folder))  # Continue numbering if folder already has images

print("Press 's' to start, 'e' to stop, and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        time.sleep(2) # Sleep for 2 seconds to allow camera to adjust
        capturing = True
        print("Started capturing...")

    elif key == ord('e'):
        capturing = False
        print("Stopped capturing.")

    elif key == ord('q'):
        break

    if capturing:
        img_path = os.path.join(output_folder, f"{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        img_count += 1
        if img_count > 10:
            capturing = False
            print("Captured 10 images, stopping.")
            time.sleep(2)

cap.release()
cv2.destroyAllWindows()
