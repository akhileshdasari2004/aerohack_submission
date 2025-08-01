import cv2

def grab_frame(output_path='frame.jpg'):
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        print(f"Trying camera index {idx}, opened: {cap.isOpened()}")
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                cv2.imwrite(output_path, frame)
                print(f"Saved frame from index {idx} to {output_path}")
                return
            else:
                print(f"Index {idx} opened but failed to read frame")
    raise RuntimeError("Cannot open any webcam; tried indices 0-2")

if __name__ == "__main__":
    grab_frame()
