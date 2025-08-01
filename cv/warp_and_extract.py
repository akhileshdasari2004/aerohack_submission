import cv2
import numpy as np
import sys

# cube color palette in BGR (approximate, tweak if needed)
PALETTE = {
    'W': np.array([230, 230, 230], dtype=np.uint8),  # white
    'Y': np.array([45, 225, 255], dtype=np.uint8),   # yellow
    'R': np.array([30, 30, 255], dtype=np.uint8),    # red
    'O': np.array([50, 130, 255], dtype=np.uint8),   # orange
    'B': np.array([255, 90, 50], dtype=np.uint8),    # blue
    'G': np.array([50, 180, 50], dtype=np.uint8),    # green
}

def order_quad(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype='float32')

def nearest_color(bgr):
    sample = np.uint8([[bgr]])
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)[0][0].astype(int)
    min_dist = float('inf')
    best = None
    for k, v in PALETTE.items():
        pal_lab = cv2.cvtColor(np.uint8([[v]]), cv2.COLOR_BGR2LAB)[0][0].astype(int)
        dist = np.linalg.norm(sample_lab - pal_lab)
        if dist < min_dist:
            min_dist = dist
            best = k
    return best

def extract_facelets(warped):
    h, w = warped.shape[:2]
    cell_h = h // 3
    cell_w = w // 3
    face = []
    debug = warped.copy()
    for i in range(3):
        row = []
        for j in range(3):
            y0, y1 = i * cell_h, (i+1)*cell_h
            x0, x1 = j * cell_w, (j+1)*cell_w
            cell = warped[y0:y1, x0:x1]
            avg_color = cv2.mean(cell)[:3]
            label = nearest_color(np.array(avg_color[::-1], dtype=np.uint8))  # cv2 uses BGR
            row.append(label)
            # draw rectangle and label for debug
            cv2.rectangle(debug, (x0, y0), (x1, y1), (0,0,0), 1)
            cv2.putText(debug, label, (x0 + 10, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        face.append(row)
    return face, debug

def main(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read", img_path)
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = None
    max_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                quad = approx
                max_area = area
    if quad is None:
        print("No quadrilateral found, falling back to full image.")
        h, w = img.shape[:2]
        pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype='float32')
    else:
        pts = order_quad(quad)
    dst_size = 300
    dst_pts = np.array([[0,0],[dst_size,0],[dst_size,dst_size],[0,dst_size]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (dst_size, dst_size))
    face, debug = extract_facelets(warped)
    face_str = ''.join([''.join(r) for r in face])
    print("Detected face (3x3):")
    for r in face:
        print(' '.join(r))
    print("Face string:", face_str)
    cv2.imwrite('warped_face.jpg', warped)
    cv2.imwrite('debug_facelets.jpg', debug)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python warp_and_extract.py frame.jpg")
    else:
        main(sys.argv[1])
