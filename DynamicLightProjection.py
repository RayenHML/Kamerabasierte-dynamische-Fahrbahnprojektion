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
from cv_bridge import CvBridge
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class ImageConverter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.image_pub = rospy.Publisher("/camera/image/processed", Image, queue_size=10)

    def callback(self, data):
        try:
            # Konvertiere ROS-Bild in OpenCV-Format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.resize(cv_image, (1600, 1200))
            original_image = cv_image.copy()
            
            # Bild zuschneiden und Maske anwenden
            image = cv_image[600:850, 300:1200]
            mask = np.zeros_like(image)
            polygons = np.array([[(0, 249), (299, 0), (599, 0), (899, 249)]], dtype=np.int32)
            cv2.fillPoly(mask, polygons, (255, 255, 255))
            image = cv2.bitwise_and(image, mask)
            
            # Kanten erkennen
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.bitwise_or(cv2.convertScaleAbs(sobelx), cv2.convertScaleAbs(sobely))
            _, binary_image = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)
            binary_image_gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
            
            # Bild verfeinern mit Morphologischer Transformation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morphologyEx = cv2.morphologyEx(binary_image_gray, cv2.MORPH_CLOSE, kernel)
            
            # Linien erkennen
            lines = cv2.HoughLinesP(morphologyEx, 1, np.pi / 180, threshold=10, minLineLength=5, maxLineGap=10)
            black_white = np.zeros((image.shape[0], image.shape[1], 3))
            
            # Linien filtern und zeichnen
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_degrees = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if not (-35 < angle_degrees < 35):
                    cv2.line(black_white, (x1, y1), (x2, y2), (255, 255, 255), 3)
            
            # Punktlisten für polynomiale Regression erstellen
            l_X, l_y = [], []
            for i in range(50, 200):
                for j in range(200, 400):
                    if int(black_white[i, j].sum()) > 0:
                        l_X.append([i])
                        l_y.append(j)
                        break
            
            # Polynomiale Regression (Grad 2) zur Kurvenanpassung
            degree = 2
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(l_X)
            model = LinearRegression()
            model.fit(X_poly, l_y)
            
            # Kurve berechnen
            x_curve_l = np.linspace(min(l_X)[0], max(l_X)[0], 100)
            y_curve_l = model.predict(poly_features.transform(x_curve_l.reshape(-1, 1)))
            
            # Ergebnisbild vorbereiten
            image_black = np.zeros((1200, 1600, 3), dtype=np.uint8) 
            points_l = np.vstack((y_curve_l + 200, x_curve_l)).T.astype(np.int32)
            cv2.polylines(image_black, [points_l], isClosed=False, color=(255, 255, 255), thickness=1)
            
            # ROS-Bild veröffentlichen
            processed_img_msg = self.bridge.cv2_to_imgmsg(image_black, "bgr8")
            self.image_pub.publish(processed_img_msg)
            
            # Debug-Anzeige (optional)
            cv2.imshow("Processed Image", image_black)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Fehler bei der Bildverarbeitung: {e}")

if __name__ == '__main__':
    rospy.init_node('image_converter', anonymous=True)
    ImageConverter()
    rospy.spin()
    cv2.destroyAllWindows()
