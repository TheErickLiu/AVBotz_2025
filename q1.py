import cv2
import numpy as np

def imagePoints(image_path):
    img = cv2.imread(image_path)
    
    # Blurring image with a kernal size of 5x5 to reduce noise
    img_blur = cv2.blur(img,(5,5))

    # Convert to grayscale
    img_grayscale = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # Filters out pixels that are below 20 on RGB scale to keep the torp board (dark)
    _ , thresholded = cv2.threshold(img_grayscale, 20, 255, cv2.THRESH_BINARY_INV)

    # Dilate kernal by five times
    kernel = np.ones((25, 25), np.uint8)
    img_crop = cv2.dilate(thresholded, kernel, iterations=1)

    # Get contours of board
    # cv2.RETR_EXTERNAL: Gives only extreme outer contours / the ones that matter
    # cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments and leaves only their end points
    contours, _ = cv2.findContours(img_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key = cv2.contourArea)

    # Fill in the contour. This is important to get contour perimeter.
    contour = cv2.convexHull(contour)
    
    # Get x and y coords of center
    x,y,w,h = cv2.boundingRect(contour)
    xCenter = int((x + x + w)/2)
    yCenter = int((y + y + h)/2)

    # Calculates closed contour perimeter
    perimeter_len = cv2.arcLength(contour, True)

    # Approximation accuracy set to +/- 3% of perimeter_len, not a dominating parameter
    corners = cv2.approxPolyDP(contour, 0.03 * perimeter_len, True)

    # Vertexes of board
    points = [point[0] for point in corners]
    if len(points) != 4:
        return (0, 0, 0)
    
    point1, point2, point3, point4 = sorted(points, key=lambda k: [k[0], k[1]])

    # calibrateCamera requires at least six points, more is better.
    if point1[1] < point2[1]:
        topleft = point1
        bottomleft = point2
    else:
        topleft = point2
        bottomleft = point1
    if point3[1] < point4[1]:
        topright= point3
        bottomright = point4
    else:
        topright = point4
        bottomright = point3

    midleft = ((topleft[0]+bottomleft[0])/2, (topleft[1]+bottomleft[1])/2)
    midright = ((topright[0]+bottomright[0])/2, (topright[1]+bottomright[1])/2) 
    midtop = ((topleft[0]+topright[0])/2, (topleft[1]+topright[1])/2) 
    midbottom = ((bottomleft[0]+bottomright[0])/2, (bottomleft[1]+bottomright[1])/2)

    return img.shape, np.array(
        [topleft, topright,
        bottomleft, bottomright,
        midleft, midright,
        midtop, midbottom,
        (xCenter, yCenter)],
        dtype= 'float32'
    )

# https://learnopencv.com/rotation-matrix-to-euler-angles/
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
        
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    # (Roll, Pitch, Yaw)
    return np.degrees(np.array([x, y, z]))
        
def run():
    expected_torp_board_alignment = [
        (252, 190, -1),
        (264, 164, -42),
        (265, 188, 4),
        (278, 188, 32)
    ]

    object_points = []
    image_points = []
    shape = None
    for i in range(len(expected_torp_board_alignment)):
        object_points.append(np.array(
            [[0, 0, 0],         # topleft
             [1, 0 , 0],        # topright
             [0, 1 , 0],        # bottomleft
             [1, 1, 0],         # bottomright
             [0, 0.5 , 0],      # midtop
             [1, 0.5 , 0],      # midbottom
             [0.5, 0 , 0],      # midleft
             [0.5, 1 , 0],      # midright
             [0.5, 0.5 , 0]],   # center
            dtype= 'float32'
        ))
        results = imagePoints(f"inputs/torpBoard{i+1}.png")
        # Images are assumed the same size for calibration
        if shape:
            assert(shape == results[0])
        else:
            shape = results[0]
        image_points.append(results[1])
    
    # shape has format (height, width) while calibrateCamera needs (width, height) as size
    _, matrix_camera, _, _, _ = cv2.calibrateCamera(object_points, image_points, shape[:2][::-1], None, None)

    for i in range(len(expected_torp_board_alignment)):
        print (f'Expected output for torpBoard_{i+1}.png: {expected_torp_board_alignment[i]}')

        distortion_coeffs = np.zeros((4,1)) # No distortion
        _ , rotation_vector, _ = cv2.solvePnP(object_points[i][0:4], image_points[i][0:4], matrix_camera, distortion_coeffs, flags=0)
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
        euler_angles = rotationMatrixToEulerAngles(rotation_matrix)

        print (f'Calculated : {image_points[i][8][0]} {image_points[i][8][1]} {euler_angles[1]}')

run()