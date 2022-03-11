from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkPlane
from vtkmodules.vtkIOImage import vtkJPEGReader
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane
from vtkmodules.vtkFiltersCore import vtkClipPolyData, vtkFeatureEdges, vtkStripper
from vtk import vtkInteractorStyleTrackballCamera, vtkLight
                
from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper, vtkDataSetMapper,
                                        vtkRenderer, vtkTexture)

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import (QMainWindow, QApplication, QDialog, QDialogButtonBox, QFrame, QSlider,
                             QPushButton, QLineEdit, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel,
                             QMessageBox, QColorDialog)
from PyQt5.QtGui import QIcon, QFont
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

from helperFunctions import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(1200, 550)
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)

        MainWindow.setWindowIcon(QIcon('optics.png'))
        MainWindow.setWindowTitle('Vtk Project : Raytracing tool')
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.colors = vtkNamedColors()
        self.createRenderers()

        self.vtkWidget = QVTKRenderWindowInteractor()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        style = vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)

        vBoxLayout = QVBoxLayout()
        pushButton = QPushButton(" Render \n raytracing")
        pushButton.setFont(QFont('Times', 12))
        self.sliderx = QSlider(Qt.Horizontal)
        self.sliderx.setValue(50)
        self.slidery = QSlider(Qt.Horizontal)
        self.slidery.setValue(100)
        self.sliderz = QSlider(Qt.Horizontal)
        self.sliderz.setValue(50)
        self.sliderCx = QSlider(Qt.Horizontal)
        self.sliderCx.setValue(50)
        self.sliderCy = QSlider(Qt.Horizontal)
        self.sliderCy.setValue(50)
        self.sliderCz = QSlider(Qt.Horizontal)
        self.sliderCz.setValue(50)
        self.textRotation = QLineEdit("")
        rotationButton = QPushButton("Rotate")
        rotationButton.setFont(QFont('Times', 8))
        configButton = QPushButton()
        configButton.setIcon(QIcon('Configuration.png'))

        vBoxLayout.addWidget(pushButton)
        vBoxLayout.addWidget(QLabel("Move Light"))
        vBoxLayout.addWidget(self.sliderx)
        vBoxLayout.addWidget(self.slidery)
        vBoxLayout.addWidget(self.sliderz)
        vBoxLayout.addWidget(QLabel("Move Camera"))
        vBoxLayout.addWidget(self.sliderCx)
        vBoxLayout.addWidget(self.sliderCy)
        vBoxLayout.addWidget(self.sliderCz)
        vBoxLayout.addWidget(QLabel("Rotate Body"))
        vBoxLayout.addWidget(self.textRotation)
        vBoxLayout.addWidget(rotationButton)
        vBoxLayout.addWidget(configButton)

        hBoxLayout = QHBoxLayout()
        hBoxLayout.addWidget(self.vtkWidget)
        hBoxLayout.addLayout(vBoxLayout)
        hBoxLayout.setStretch(0, 1)

        pushButton.clicked.connect(self.rayTrancingRender)
        self.sliderx.valueChanged.connect(self.moveAgentx)
        self.slidery.valueChanged.connect(self.moveAgenty)
        self.sliderz.valueChanged.connect(self.moveAgentz)
        self.sliderCx.valueChanged.connect(self.moveCamerax)
        self.sliderCy.valueChanged.connect(self.moveCameray)
        self.sliderCz.valueChanged.connect(self.moveCameraz)
        rotationButton.clicked.connect(self.rotateBody)
        configButton.clicked.connect(self.showConfigWindow)
        
        self.frame.setLayout(hBoxLayout)
        self.setCentralWidget(self.frame)

        self.vtkWidget.GetRenderWindow().Render()
        self.show()
        iren.Initialize()
        iren.Start()
    
    def showConfigWindow(self):
        def getColor():
            color = QColorDialog.getColor()
            if color:
                self.backgroundColor = [c/ 255 for c in list(color.getRgb())[:3]]
                self.ren.SetBackground(self.backgroundColor)
                self.vtkWidget.GetRenderWindow().Render()

        def saveConfigs(width_text, height_text, depth_text):
            if width_text.text() and height_text.text() and depth_text.text():
                self.width = int(width_text.text())
                self.height = int(height_text.text())
                self.max_depth = int(depth_text.text())
                
            else:
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setWindowTitle("Warning")
                msgBox.setText("Be sure to fill out all the fields!")
                msgBox.exec_()

        win = QDialog()
        win.setWindowTitle("Configurations")
        win.setWindowIcon(QIcon('configuration.png'))

        label1 = QLabel("Image resolution (width/height)")
        width_text = QLineEdit(f"{self.width}")
        height_text = QLineEdit(f"{self.height}")
        label2 = QLabel("Number of reflexions")
        depth_text = QLineEdit(f"{self.max_depth}")
        colorButton = QPushButton(" Background \n  color")

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, QtCore.Qt.Horizontal)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0, 1, 2)
        layout.addWidget(width_text, 1, 0)
        layout.addWidget(height_text, 1, 1)
        layout.addWidget(label2, 2, 0)
        layout.addWidget(depth_text, 3, 0)
        layout.addWidget(colorButton, 2, 1, 2, 1)
        layout.addWidget(buttonBox, 4, 1)

        colorButton.clicked.connect(getColor)
        buttonBox.accepted.connect(win.accept)
        buttonBox.accepted.connect(lambda: saveConfigs(width_text, height_text, depth_text))

        win.setLayout(layout)
        win.exec_()

    
    def createRenderers(self):
        self.ren = vtkRenderer()
        self.mainActors = []
        self.objects = []
        self.actors = []
        
        bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
        self.colors.SetColor("ivory_black", *bkg)
        self.backgroundColor = list(self.colors.GetColor3d("ivory_black"))

        self.createCamera()
        self.addActors()

        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(0.7)
    
    def createCamera(self):
        self.width = 100
        self.height = 100
        self.max_depth = 2

        self.camera = np.array([0, 1, 4])
        self.initCamera = np.array([0, 1, 4])
        self.ratio = float(self.width) / self.height
        self.screen_size = 0.5
        self.distScreen = 1
        self.screen = coordsScreen(self.screen_size, self.camera, self.ratio)
        self.sunLight = {'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 
                        'specular': np.array([1, 1, 1])}
        
        self.cameraSource = vtkSphereSource()
        self.cameraSource.SetCenter(l2n(self.camera))
        self.cameraSource.SetRadius(0.1)
        self.cameraSource.SetThetaResolution(10)
        self.cameraSource.SetPhiResolution(10)
        mapperCamera = vtkPolyDataMapper()
        mapperCamera.SetInputConnection(self.cameraSource.GetOutputPort())
        self.actorCamera = vtkActor()
        self.actorCamera.SetMapper(mapperCamera)
        self.actorCamera.GetProperty().SetColor([1, 0, 0])

        self.ren.AddActor(self.actorCamera)

        self.screenSource = vtkCubeSource()
        self.screenSource.SetXLength(self.screen[2] - self.screen[0])
        self.screenSource.SetYLength(self.screen[1] - self.screen[3])
        self.screenSource.SetZLength(0.1)
        self.screenSource.SetCenter(l2n(self.camera - (0, 0, self.distScreen)))
        outline = vtkOutlineFilter()
        outline.SetInputConnection(self.screenSource.GetOutputPort())
        outlineMapper = vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        self.screenActor = vtkActor()
        self.screenActor.SetMapper(outlineMapper)

        self.ren.AddActor(self.screenActor)

    def addActors(self):
        configs = [ {"image_file": "Logo_HPC_AI.jpg", "cutPlane": [0.6, 0, 0], "angle": 0},
                    {"image_file": "cloud.jpg", "cutPlane": [-0.6, 0, 0], "angle": 90},
                    {"image_file": "hand-7014643_1920.jpg", "cutPlane": [0, 0, 0.6], "angle": 180},
                    {"image_file": "LogoMines.jpg", "cutPlane": [0, 0, -0.6], "angle": 270},
        ]

        for config in configs:
            reader = vtkJPEGReader()
            reader.SetFileName(config["image_file"])
            texture = vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())

            cutplaneRen = vtkPlane()
            cutplaneRen.SetOrigin(*config["cutPlane"])
            cutplaneRen.SetNormal(*[scalar/6*10 for scalar in config["cutPlane"]])

            cubeRen = vtkCubeSource()
            cubeRen.SetXLength(0.8)
            cubeRen.SetYLength(0.8)
            cubeRen.SetZLength(0.8)
            cubeRen.SetCenter(*[scalar/2 for scalar in config["cutPlane"]])  
            clipperRen = vtkClipPolyData()
            clipperRen.SetInputConnection(cubeRen.GetOutputPort())
            clipperRen.GenerateClippedOutputOn()
            clipperRen.SetClipFunction(cutplaneRen)
            clipperRen.SetValue(0)
            clipperRen.Update()
            self.objects.append(clipperRen)
            clipperMapperRen = vtkPolyDataMapper()
            clipperMapperRen.SetInputConnection(clipperRen.GetOutputPort())
            clippedRenActor = vtkActor()
            clippedRenActor.SetMapper(clipperMapperRen)
            clippedRenActor.GetProperty().SetColor([0.9, 0.9, 0.9])
            clippedRenActor.GetProperty().SetAmbientColor([0.1, 0.1, 0.1])
            clippedRenActor.GetProperty().SetDiffuseColor([0.9, 0.9, 0.9])
            clippedRenActor.GetProperty().SetSpecularColor([1, 1, 1])
            self.actors.append(clippedRenActor)

            cutplaneImage = vtkPlane()
            cutplaneImage.SetOrigin(0.3, 0, 0)
            cutplaneImage.SetNormal(1, 0, 0)
            cubeImage = vtkCubeSource()
            cubeImage.SetXLength(0.8)
            cubeImage.SetYLength(0.8)
            cubeImage.SetZLength(0.8)
            cubeImage.SetCenter(0.3, 0, 0)
            clipperImage = vtkClipPolyData()
            clipperImage.SetInputConnection(cubeImage.GetOutputPort())
            clipperImage.GenerateClippedOutputOn()
            clipperImage.SetClipFunction(cutplaneImage)
            clipperImage.SetValue(0)
            clipperImage.Update()
            mapperImage = vtkPolyDataMapper()
            mapperImage.SetInputConnection(clipperImage.GetOutputPort())
            clippedImageActor = vtkActor()
            clippedImageActor.SetMapper(mapperImage)
            clippedImageActor.SetTexture(texture)
            clippedImageActor.GetProperty().SetColor([0.9, 0.9, 0.9])
            clippedImageActor.RotateY(config["angle"])

            boundaryEdges = vtkFeatureEdges()
            boundaryEdges.SetInputConnection(clipperRen.GetOutputPort())
            boundaryEdges.BoundaryEdgesOn()
            boundaryEdges.FeatureEdgesOff()
            boundaryEdges.NonManifoldEdgesOff()
            boundaryEdges.ManifoldEdgesOff()

            boundaryStrips = vtkStripper()
            boundaryStrips.SetInputConnection(boundaryEdges.GetOutputPort())
            boundaryStrips.Update()

            # Change the polylines into polygons
            boundaryPoly = vtkPolyData()
            boundaryPoly.SetPoints(boundaryStrips.GetOutput().GetPoints())
            boundaryPoly.SetPolys(boundaryStrips.GetOutput().GetLines())

            boundaryMapper = vtkPolyDataMapper()
            boundaryMapper.SetInputData(boundaryPoly)

            boundaryActor = vtkActor()
            boundaryActor.SetMapper(boundaryMapper)

            self.ren.AddActor(clippedImageActor)
            self.ren.AddActor(boundaryActor)
            self.mainActors.append(clippedRenActor)
            self.mainActors.append(boundaryActor)

        mainCube =vtkCubeSource()
        mainCube.SetXLength(1.2)
        mainCube.SetYLength(1.2)
        mainCube.SetZLength(1.2)
        mainCube.SetCenter((0, 0, 0))
        self.mainCube = mainCube
        self.objects.append(mainCube)

        mainCubeMapper = vtkDataSetMapper()
        mainCubeMapper.SetInputConnection(mainCube.GetOutputPort())
        mainCubeActor = vtkActor()
        mainCubeActor.SetMapper(mainCubeMapper)
        mainCubeActor.GetProperty().SetColor([0.7, 0, 0])
        mainCubeActor.GetProperty().SetAmbientColor([0.1, 0, 0])
        mainCubeActor.GetProperty().SetDiffuseColor([0.7, 0, 0])
        mainCubeActor.GetProperty().SetSpecularColor([1, 1, 1])
        mainCubeActor.GetProperty().SetDiffuse(1)
        mainCubeActor.GetProperty().SetInterpolationToPhong()
        # mainCubeActor.GetProperty().SetRoughness(0.1)
        # mainCubeActor.GetProperty().SetMetallic(1)
        self.mainCubeActor = mainCubeActor
        self.mainActors.append(mainCubeActor)
        self.actors.append(mainCubeActor)

        self.cubeRef = vtkCubeSource()
        self.cubeRef.SetXLength(6)
        self.cubeRef.SetYLength(3.4)
        self.cubeRef.SetZLength(8)
        self.cubeRef.SetCenter((0, 1.1, 1.5))
        self.objects.append(self.cubeRef)

        outline = vtkOutlineFilter()
        outline.SetInputConnection(self.cubeRef.GetOutputPort())
        outlineMapper = vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtkActor()
        outlineActor.SetMapper(outlineMapper)

        cubeRefMapper = vtkDataSetMapper()
        cubeRefMapper.SetInputConnection(self.cubeRef.GetOutputPort())
        cubeRefActor = vtkActor()
        cubeRefActor.SetMapper(cubeRefMapper)
        cubeRefActor.GetProperty().SetColor([0.6, 0.6, 0.6])
        cubeRefActor.GetProperty().SetAmbientColor([0.1, 0.1, 0.1])
        cubeRefActor.GetProperty().SetDiffuseColor([0.6, 0.6, 0.6])
        cubeRefActor.GetProperty().SetSpecularColor([1, 1, 1])
        self.actors.append(cubeRefActor)

        self.createSunActor()

        self.ren.AddActor(outlineActor)
        self.ren.AddActor(mainCubeActor)
        self.ren.AddActor(clippedImageActor)
        self.ren.AddActor(self.sunActor)
        self.ren.SetBackground(self.backgroundColor)

        #   This adds light to the scene, but I preferred not to use it.
        # self.createLight()
        # self.ren.AddLight(self.light)

    def createSunActor(self):
        ResolutionSun = 10
        sun = vtkSphereSource()
        sun.SetCenter(0.0, 2.7, 1.4)
        sun.SetRadius(0.1)
        sun.SetThetaResolution(ResolutionSun)
        sun.SetPhiResolution(ResolutionSun)
        self.sun = sun
        mapperSun = vtkPolyDataMapper()
        mapperSun.SetInputConnection(sun.GetOutputPort())
        self.sunActor = vtkActor()
        self.sunActor.SetMapper(mapperSun)
        self.sunActor.GetProperty().SetColor([1.0, 1.0, 0.0]) 

    
    def createLight(self):
        self.light = vtkLight()
        self.light.SetAmbientColor([1.0, 1.0, 0.0])
        self.light.SetConeAngle(180)
        self.light.SetIntensity(100)
        self.light.SetPosition(self.sunActor.GetCenter())
        self.light.SetFocalPoint(self.mainCube.GetCenter())
        self.light.SetPositional(True)
    
    def moveAgentx(self):
        for agent in [self.sunActor]:
            agent.SetPosition([(self.sliderx.value() - 50) / 18, 
                    (self.slidery.value() - 100) / 33, (self.sliderz.value() - 50) / 14])

        self.vtkWidget.GetRenderWindow().Render()
    
    def moveAgenty(self):
        for agent in [self.sunActor]:
            agent.SetPosition([(self.sliderx.value() - 50) / 18, 
                    (self.slidery.value() - 100) / 33, (self.sliderz.value() - 50) / 14])

        self.vtkWidget.GetRenderWindow().Render()
    
    def moveAgentz(self):
        for agent in [self.sunActor]:
            agent.SetPosition([(self.sliderx.value() - 50) / 18, 
                    (self.slidery.value() - 100) / 33, (self.sliderz.value() - 50) / 14])
        self.vtkWidget.GetRenderWindow().Render()
    
    def moveCamerax(self):
        for agent in [self.actorCamera, self.screenActor]:
            agent.SetPosition([(self.sliderCx.value() - 50) / 18, 
                    (self.sliderCy.value() - 50) / 50, (self.sliderCz.value() - 50) / 50])

        self.camera = self.initCamera + l2n(self.actorCamera.GetPosition())
        self.screen = coordsScreen(self.screen_size, self.camera, self.ratio)
        self.vtkWidget.GetRenderWindow().Render()
    
    def moveCameray(self):
        for agent in [self.actorCamera, self.screenActor]:
            agent.SetPosition([(self.sliderCx.value() - 50) / 18, 
                    (self.sliderCy.value() - 50) / 50, (self.sliderCz.value() - 50) / 50])
        
        self.camera = self.initCamera + l2n(self.actorCamera.GetPosition())
        self.screen = coordsScreen(self.screen_size, self.camera, self.ratio)
        self.vtkWidget.GetRenderWindow().Render()
    
    def moveCameraz(self):
        for agent in [self.actorCamera, self.screenActor]:
            agent.SetPosition([(self.sliderCx.value() - 50) / 18, 
                    (self.sliderCy.value() - 50) / 50, (self.sliderCz.value() - 50) / 50])
        
        self.camera = self.initCamera + l2n(self.actorCamera.GetPosition())
        self.screen = coordsScreen(self.screen_size, self.camera, self.ratio)
        self.vtkWidget.GetRenderWindow().Render()

    
    def rotateBody(self):
        if self.textRotation.text() and self.textRotation.text()!="-":
            angle = float(self.textRotation.text())
            for actor in self.mainActors:
                actor.RotateY(angle)
            self.vtkWidget.GetRenderWindow().Render()

    def nearest_intersected_object(self, objects, obbs, origin, direction):
        distances = []
        cellIds = []
        for obb in obbs:
            pTarget = origin + 40*direction
            if isHit(obb, origin, pTarget): 
                pointsInter, cellIdsInter = GetIntersect(obb, origin, pTarget)
                firstPoint = pointsInter[0]
                cellId = cellIdsInter[0]
                distance = np.linalg.norm(l2n(firstPoint) - origin)
                distances.append(distance)
                cellIds.append(cellId)
                # addLine(self.ren, origin, firstPoint)
            else:
                distances.append(None)
                cellIds.append(None)

        nearest_object = None
        cellId = None
        min_distance = np.inf
        for index, distance in enumerate(distances):
            if distance and distance < min_distance:
                min_distance = distance
                nearest_object = objects[index]
                cellId = cellIds[index]

        return nearest_object, min_distance, cellId


    def rayTrancingRender(self):
        obbs = getObbs(self.objects)

        image = np.zeros((self.height, self.width, 3))
        for i, y in enumerate(np.linspace(self.screen[1], self.screen[3], self.height)):
            for j, x in enumerate(np.linspace(self.screen[0], self.screen[2], self.width)):
                # screen is on origin
                pixel = np.array([x, y, self.camera[2] - self.distScreen])
                origin = self.camera
                direction = normalize(pixel - origin)
                # addLine(self.ren, origin, pixel, color=[0.0, 0.0, 1.0])

                color = np.zeros((3))
                reflection = 1

                for k in range(self.max_depth):
                    # check for intersections
                    nearest_object, min_distance, cellId = self.nearest_intersected_object(
                                                                self.objects, obbs, origin, direction)
                    if nearest_object is None:
                        break
              
                    intersection = origin + min_distance * direction
                    # addLine(self.ren, origin, intersection, color=[0.0, 0.0, 1.0])
                    normal_to_surface = calcNormals(nearest_object, cellId, direction)

                    
                    shifted_point = intersection + 1e-5 * normal_to_surface
                    addPoint(self.ren, shifted_point, radius=0.05, color=[0.0, 0.0, 0.0])
                    intersection_to_light = normalize(l2n(self.sunActor.GetCenter()) - shifted_point)

                    _, min_distance, _ = self.nearest_intersected_object(self.objects, obbs,
                                                            shifted_point, intersection_to_light)

                    intersection_to_light_distance = np.linalg.norm(
                                                l2n(self.sunActor.GetCenter()) - intersection)
                    is_shadowed = min_distance < intersection_to_light_distance

                    if is_shadowed:
                        break
                    
                    illumination = np.zeros((3))

                    index = self.objects.index(nearest_object)
                    actor = self.actors[index]
                    prop = actor.GetProperty()
                    shininess = 100
                    reflection_objects = 0.5 if nearest_object == self.cubeRef else 0.05

                    # ambiant
                    illumination += l2n(prop.GetAmbientColor()) * self.sunLight['ambient']

                    # diffuse
                    illumination += (l2n(prop.GetDiffuseColor()) * self.sunLight['diffuse'] *
                                     np.dot(intersection_to_light, normal_to_surface))

                    # specular
                    intersection_to_camera = normalize(self.camera - intersection)
                    H = normalize(intersection_to_light + intersection_to_camera)
                    illumination += (l2n(prop.GetSpecularColor()) * self.sunLight['specular'] *  
                                    np.dot(normal_to_surface, H) ** (shininess / 4))

                    # reflection
                    color += reflection * illumination
                    reflection *= reflection_objects

                    origin = shifted_point
                    direction = reflected(direction, normal_to_surface)

                image[i, j] = np.clip(color, 0, 1)

        image = cv2.resize(image, dsize=(320, int(320/self.ratio)), interpolation=cv2.INTER_CUBIC)
        image -= image.min()
        image /= image.max()

        plt.imsave('RayTracing.png', image)
        showImage()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())