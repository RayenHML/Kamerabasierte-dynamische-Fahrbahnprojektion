#!/usr/bin/env python3

"""
Script zur Verarbeitung und Veröffentlichung von Kamerabildern mit ROS.

Dieses Skript:
1. Abonniert ein ROS-Topic, das rohe Kamerabilder empfängt.
2. Verarbeitet die Bilder durch Zuschneiden, Maskieren, Kanten- und Linienerkennung.
3. Berechnet und zeichnet Kurven basierend auf den erkannten Linien.
4. Berechnet den Krümmungsradius der Kurven.
5. Veröffentlicht das verarbeitete Bild auf einem ROS-Topic.

Abhängigkeiten:
- rospy: `pip install rospy`
- sensor_msgs: ROS-Paket
- cv_bridge: ROS-Paket für die Konvertierung zwischen ROS-Bildnachrichten und OpenCV-Bildern
- cv2 (OpenCV): `pip install opencv-python-headless`
- numpy: `pip install numpy`
- scikit-learn: `pip install scikit-learn`

Verwendung:
1. Stellen Sie sicher, dass ROS installiert und konfiguriert ist.
2. Passen Sie die abonnierten und veröffentlichten ROS-Topics an.
3. Starten Sie den ROS-Knoten mit `rosrun <package_name> <script_name>.py`.
4. Das Skript empfängt und verarbeitet Kamerabilder und veröffentlicht die Ergebnisse.

Autor: Hosseini, Seyed Amirhossein
Datum: 14. Mai 2024
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Schwellwert für den Krümmungsradius
RADIUS_THRESHOLD = 0.002

def curvature_radius(x, a, b):
    # Berechnet den Krümmungsradius an einer Stelle x anhand der Koeffizienten a und b.
    numerator = abs(2 * a)
    denominator = (1 + (2 * a * x + b)**2)**(1.5)
    return numerator / denominator

def process_side(image, side='left'):
    """
    Extrahiert Koordinaten und passt eine quadratische Regression an.
    Für die linke Seite wird im Bereich j von 200 bis 400 gesucht,
    für die rechte Seite im Bereich j von 899 bis 500.
    """
    coords = []
    if side == 'left':
        j_start, j_bound, j_step = 200, 400, 1
    else:
        j_start, j_bound, j_step = 899, 500, -1

    for i in range(50, 150, 5):
        j = j_start
        if side == 'left':
            while j < j_bound and int(image[i, j].sum()) == 0:
                j += j_step
            if j < j_bound:
                coords.append((i, j))
        else:
            while j > j_bound and int(image[i, j].sum()) == 0:
                j += j_step
            if j > j_bound:
                coords.append((i, j))

    if not coords:
        return None

    l_X = [[coord[0]] for coord in coords]
    l_y = [coord[1] for coord in coords]

    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(l_X)
    model = LinearRegression()
    model.fit(X_poly, l_y)

    a = model.coef_[2]  # Koeffizient für x^2
    b = model.coef_[1]  # Koeffizient für x
    c = model.intercept_

    x_values = np.linspace(min(l_X)[0], max(l_X)[0], 300)
    radii = [curvature_radius(x, a, b) for x in x_values]
    min_radius = min(radii)

    # Erzeuge Kurvendaten zur Darstellung
    x_curve = np.linspace(10, 150, num=150)
    y_curve = model.predict(poly_features.transform(x_curve.reshape(-1, 1)))

    return {
        'model': model,
        'poly_features': poly_features,
        'a': a,
        'b': b,
        'c': c,
        'min_radius': min_radius,
        'x_curve': x_curve,
        'y_curve': y_curve
    }

class ImageConverter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback, queue_size=1)
        self.image_pub = rospy.Publisher("/camera/image/processed", Image, queue_size=1)
    
    def callback(self, data):
        try:
            # Konvertiere ROS-Bild in OpenCV-Format und skaliere es
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.resize(cv_image, (1600, 1200))
            original_image = cv_image.copy()
            
            # Zuschneiden des interessierenden Bildbereichs
            image = cv_image[600:850, 300:1200, :]
            
            # Anwenden einer Polygonmaske
            polygons = np.array([[(0, 249), (299, 0), (599, 0), (899, 249)]], dtype=np.int32)
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, polygons, (255, 255, 255))
            image = cv2.bitwise_and(image, mask)
            
            # Kantenerkennung mittels Sobel-Operatoren
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.bitwise_or(cv2.convertScaleAbs(sobelx), cv2.convertScaleAbs(sobely))
            _, binary_image = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)
            cv2.polylines(binary_image, polygons, isClosed=True, color=(0, 0, 0), thickness=5)
            
            binary_image_gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary_image_gray, kernel, iterations=1)
            median_filtered = cv2.medianBlur(dilated, 3)
            kernel_close = np.ones((3, 3), np.uint8)
            morphed = cv2.morphologyEx(median_filtered, cv2.MORPH_CLOSE, kernel_close)
            
            # Linienerkennung mittels Hough-Transformation
            lines = cv2.HoughLinesP(morphed, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=10)
            black_white = np.zeros_like(image)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle_degrees = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if not (-35 < angle_degrees < 35):
                        cv2.line(black_white, (x1, y1), (x2, y2), (255, 255, 255), 3)
            
            # Verarbeitung der linken und rechten Seite
            left_result = process_side(black_white, side='left')
            right_result = process_side(black_white, side='right')
            
            if (left_result is not None and right_result is not None and 
                left_result['min_radius'] < RADIUS_THRESHOLD and right_result['min_radius'] < RADIUS_THRESHOLD):
                
                # Anpassen der Kurvendaten an den Bildausschnitt
                new_y_curve_left = left_result['y_curve'] + 300
                new_x_curve_left = left_result['x_curve'] + 800

                new_y_curve_right = right_result['y_curve'] + 400
                new_x_curve_right = right_result['x_curve'] + 800

                y_min = min(np.min(new_y_curve_left), np.min(new_y_curve_right))
                y_max = max(np.max(new_y_curve_left), np.max(new_y_curve_right))

                adjusted_y_left = np.clip(new_y_curve_left, y_min, y_max)
                adjusted_y_right = np.clip(new_y_curve_right, y_min, y_max)

                # Erstellen der Punkte für das Polygon
                points_left = np.vstack((adjusted_y_left, new_x_curve_left)).T.astype(np.int32)
                points_right = np.vstack((adjusted_y_right, new_x_curve_right)).T.astype(np.int32)

                # Erstellen des finalen schwarzen Bildes und Zeichnen des Polygons
                image_black = np.zeros((1200, 1600, 3), dtype=np.uint8)
                poly_points = np.vstack([points_left, points_right[::-1]])
                cv2.polylines(image_black, [points_left], isClosed=False, color=(255, 255, 255), thickness=1)
                cv2.polylines(image_black, [points_right], isClosed=False, color=(255, 255, 255), thickness=1)
                cv2.fillPoly(image_black, [poly_points], color=(255, 255, 255))

                # Auswahl eines Bildausschnitts und Skalierung
                image_black_result = image_black[800:1000, 300:1200, :]
                height, width = image_black_result.shape[:2]
                processed_image = cv2.resize(image_black_result, (2 * width, 5 * height), interpolation=cv2.INTER_LINEAR)
            else:
                rospy.loginfo("Mindestens einer der Radien ist nicht kleiner als 0.002")
                width, height = 1800, 1000
                processed_image = np.zeros((height, width, 3), dtype=np.uint8)
                points = np.array([[600, 300], [1000, 300], [1500, 1200], [300, 1200]])
                cv2.fillPoly(processed_image, [points], color=(255, 255, 255))
            
            # ROS-Bild veröffentlichen
            processed_img_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            self.image_pub.publish(processed_img_msg)
            
            # Debug-Anzeige (optional)
            cv2.imshow("Processed Image", processed_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Fehler bei der Bildverarbeitung: %s", e)

if __name__ == '__main__':
    rospy.init_node('image_converter', anonymous=True)
    converter = ImageConverter()
    rospy.spin()
    cv2.destroyAllWindows()
