# Kamerabasierte-dynamische-Fahrbahnprojektion

## Überblick





<img src="https://github.com/RayenHML/Kamerabasierte-dynamische-Fahrbahnprojektion/blob/main/Bilder/002.png" width="400" height="400">

Dieses Repository enthält das Python-Skript, das eine Bildverarbeitung zur Erkennung der Fahrspuren und anschließlich Projektion der Lichtprojektionen auf der Fahrbahn in ROS durchführt.

Dieses Skript führt die folgenden Schritte aus:
1. Abonnieren eines ROS-Topics, das rohe Kamerabilder empfängt.
2. Verarbeitung der empfangenen Bilder durch Zuschneiden, Maskieren, Kanten- und Linienerkennung.
3. Berechnung und Zeichnung von Kurven basierend auf den erkannten Linien.
4. Berechnung des Krümmungsradius der Kurven.
5. Veröffentlichung des verarbeiteten Bildes auf einem ROS-Topic.

# Inhaltverzeichnis
- [Überblick](#überblick)
- [Inhaltverzeichnis](#inhaltverzeichnis)
- [Installation](#installation)
- [Aufbau des Coes](#aufbau-des-codes)
- [Hauptbestandteile des Codes](#hauptbestandteile-des-codes)
  
## Installation

**Abhängigkeiten:**

Das Skript benötigt einige ROS-Pakete und Python-Bibliotheken. Installiere diese mit:

- rospy: `pip install rospy`
- sensor_msgs: ROS-Paket
- cv_bridge: ROS-Paket für die Konvertierung zwischen ROS-Bildnachrichten und OpenCV-Bildern `sudo apt install ros-$(rosversion -d)-cv-bridge`
- cv2 (OpenCV): `pip install opencv-python-headless`
- numpy: `pip install numpy`
- scikit-learn: `pip install scikit-learn`

**Hinweise:**
1. Stellen Sie sicher, dass ROS installiert und konfiguriert ist.
2. Ändern Sie gegebenenfalls die abonnierten und veröffentlichten ROS-Topics.
3. Starten Sie den ROS-Knoten mit dem Befehl `rosrun <package_name> <script_name>.py`.
4. Das Skript geht davon aus, dass die Bildnachrichten im Format "bgr8" vorliegen.
5. Das Skript veröffentlicht sowohl das Originalbild als auch das verarbeitete Bild auf separaten ROS-Topics.
6. Das Skript empfängt und verarbeitet Bilder von der Kamera und veröffentlicht die verarbeiteten Bilder auf einem ROS-Topic.

## Ausführung


**ROS Master starten**

ROS benötigt einen zentralen Master, um die Kommunikation zwischen Nodes zu ermöglichen. Starte diesen mit:

`roscore`

Darauf achten: Das Terminal sollte geöffnet bleiben, damit der Master weiterarbeiten kann.

**Kamera-Node starten**

Das Skript erwartet Bilddaten von einer Kamera. Um Kamera zu starten und Bilder zu empfangen, wird folgende Code ausgeführt:

`rosrun kamera_package pub_img_node`

**ROS-Umgebung einrichten**

In einem neuen Terminal muss die ROS-Umgebung geladen werden, damit das Skript richtig auf ROS zugreifen kann:

`source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
source devel/setup.bash`

**Skript ausführen**

Nun kann das Skript gestartet werden, das die Bilder verarbeitet und veröffentlicht:

`rosrun kamera_package image_converter.py`

## Aufbau des Codes

Das Skript basiert auf der ImageConverter-Klasse, die ROS-Nachrichten empfängt, Bilder verarbeitet und die Ergebnisse als neue ROS-Nachrichten veröffentlicht. Es enthält:

Einen Subscriber, der Bilder vom ``Topic /camera/image_raw`` empfängt.

Einen Publisher, der das verarbeitete Bild auf ``/camera/image/processed`` veröffentlicht.

Eine Callback-Funktion, die die Bildverarbeitung durchführt.

## Hauptbestandteile des Codes

Das Skript ``lane_detection.py`` verarbeitet Bilder einer Kamera und veröffentlicht die Ergebnisse über ROS.

**Initialisierung der ROS-Knoten und Topics**
```bash
class ImageConverter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.image_pub = rospy.Publisher("/camera/image/processed", Image, queue_size=10)
```
Hier wird die CvBridge-Klasse verwendet, um ROS-Images in OpenCV-Formate zu konvertieren. Das Skript abonniert das Kamerabild und erstellt ein Publisher-Topic für das verarbeitete Bild.

**Empfang und Vorverarbeitung des Bildes**
```bash
def callback(self, data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv_image = cv2.resize(cv_image, (1600, 1200))
        original_image = cv_image.copy()
        image = cv_image[600:850, 300:1200]
```
Das eingehende Bild wird auf eine feste Größe skaliert und ein relevanter Bereich extrahiert.

**Kanten- und Linienerkennung mit Sobel-Operator & Hough-Transformation**

```bash
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.bitwise_or(cv2.convertScaleAbs(sobelx), cv2.convertScaleAbs(sobely))
_, binary_image = cv2.threshold(sobel_combined, 100, 255, cv2.THRESH_BINARY)
```
Hier wird das Bild mit dem Sobel-Operator gefiltert, um Kanten zu erkennen. Danach wird eine binäre Maske erstellt.

**Erkennung von Linien mit Hough-Transformation**
```bash
lines = cv2.HoughLinesP(morphologyEx, 1, np.pi / 180, threshold=10, minLineLength=5, maxLineGap=10)
black_white = np.zeros((image.shape[0], image.shape[1], 3))

for line in lines:
    x1, y1, x2, y2 = line[0]
    dx = x2 - x1
    dy = y2 - y1
    angle_radians = np.arctan2(dy, dx)
    angle_degrees = np.degrees(angle_radians)
    if not (-35 < angle_degrees < 35):
        cv2.line(black_white, (x1, y1), (x2, y2), (255, 255, 255), 3)
```
Diese Methode erkennt Linien im Bild. Linien außerhalb eines bestimmten Winkels werden ausgefiltert. Anschließend werden fast horizontale Linien herausgefiltert, sodass nur steile Linien übrig bleiben. Die gefilterten Linien werden auf einem schwarzen Hintergrund weiß gezeichnet. 

**Polynomiale Regression zur Anpassung von Kurven**
```bash
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(l_X)
model = LinearRegression()
model.fit(X_poly, l_y)
y_curve_l = model.predict(poly_features.transform(x_curve_l.reshape(-1, 1)))
```
Dieser Code verwendet eine polynomiale Regression (Grad 2), um eine Kurve an gegebene Datenpunkte anzupassen. Zuerst werden die Eingangsmerkmale mit `PolynomialFeatures` transformiert, dann trainiert ein lineares Regressionsmodell die angepasste Funktion. Schließlich wird die Kurve mit den vorhergesagten Werten erstellt.

**Erstellung und Veröffentlichung des bearbeiteten Bildes**
```bash
processed_img_msg = self.bridge.cv2_to_imgmsg(original_image, "bgr8")
self.image_pub.publish(processed_img_msg)
```
Das bearbeitete Bild wird zurück in eine ROS-Message umgewandelt und veröffentlicht.

### Topics
Subscriber: ``/camera/image_raw`` (empfängt Bilder von der Kamera)

Publisher: ``/camera/image/processed`` (veröffentlicht das verarbeitete Bild)




