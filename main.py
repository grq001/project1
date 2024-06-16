import cv2
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from ultralytics import YOLO
from kivy.core.window import Window
from kivy.uix.image import Image as KivyImage
from kivy.clock import Clock
import mediapipe as mp
import math
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture

class reconocimientomanos:
    def __init__(self, mode=False, max=6, Confd=0.5, Confs=0.5):
        self.mode = mode
        self.max = max
        self.Confd = Confd
        self.Confs = Confs
        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(static_image_mode=self.mode, max_num_hands=self.max, min_detection_confidence=self.Confd, min_tracking_confidence=self.Confs)
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    def encontrar(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(imgcolor)
        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpmanos.HAND_CONNECTIONS)
        return frame

    def posicion(self, frame, ManoNum=0, dibujar=True):
        xlista = []
        ylista = []
        bbox = []
        self.lista = []
        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[ManoNum]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])
                if dibujar:
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), cv2.FILLED)
            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax
            if dibujar:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 0, 255), 2)
        return self.lista, bbox
    
    def arriba(self):
        dedos = []
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)
        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)
        return dedos
    
    def distancia(self, p1, p2, frame, dibujar=True, r=15, t=3):
        x1, y1 = self.lista[p1][1:]
        x2, y2 = self.lista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if dibujar:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)            
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED) 
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED) 
        lenght = math.hypot(x2 - x1, y2 - y1)
        return lenght, frame, [x1, y1, x2, y2, cx, cy]

class SelectModelScreen(Screen):
    def open_filechooser(self):
        content = FileChooserPopup(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Seleccionar Modelo", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        try:
            global model
            model = YOLO(filename[0])
            self.manager.current = 'detection'
            self.dismiss_popup()
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

    def dismiss_popup(self):
        self._popup.dismiss()

class FileChooserPopup(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class DetectionScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = reconocimientomanos()
        self.capture = None

    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, 1280)
        self.capture.set(4, 720)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS
    
    def on_leave(self):
        if self.capture:
            self.capture.release()
        self.capture = None

    def update(self, dt):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                results = model.predict(frame, imgsz=640, conf=0.5)
                frame = self.detector.encontrar(frame)
                lista, bbox = self.detector.posicion(frame)

                if len(results) > 0:
                    for res in results:
                        print("Detectar Basura")
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        data = frame[y1:y2, x1:x2]
                    annotated_frames = results[0].plot()
                    buf = cv2.flip(annotated_frames, 0).tostring()
                    texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                    self.ids.image.texture = texture
                else:
                    buf = cv2.flip(frame, 0).tostring()
                    texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                    self.ids.image.texture = texture

class ScreenManagement(ScreenManager):
    pass

class MyApp(App):
    def build(self):
        Window.size = (400, 200)
        sm = ScreenManagement()
        sm.add_widget(SelectModelScreen(name='select_model'))
        sm.add_widget(DetectionScreen(name='detection'))
        return sm

if __name__ == '__main__':
    MyApp().run()
