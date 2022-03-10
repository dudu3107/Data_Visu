from tkinter.ttk import Style
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR, vtkPoints
from vtkmodules.vtkCommonDataModel import (vtkImageData, vtkCellArray, vtkPolyData, 
                                          vtkPolyLine, vtkOctreePointLocator, vtkPlane)
from vtkmodules.vtkFiltersGeometry import vtkImageDataGeometryFilter
from vtkmodules.vtkIOImage import vtkPNGWriter, vtkPNGReader, vtkJPEGReader
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkCylinderSource, vtkSphereSource
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkFiltersGeneral import vtkClipDataSet
from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane, vtkTextureMapToSphere
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkInteractionStyle
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkClipPolyData, vtkFeatureEdges, vtkStripper
from vtk import (vtkInteractorStyleTrackballCamera, vtkObject, vtkLight, vtkCellCenters, vtkCellCenters,
                vtkGlyph3D, vtkArrowSource, vtkPolyDataNormals, vtkOBBTree, vtkIdList, vtkLineSource,
                VTK_UNSIGNED_CHAR)
                
from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper, vtkRenderWindow, vtkDataSetMapper,
                                        vtkRenderWindowInteractor, vtkRenderer, vtkTexture, vtkProperty,
                                        vtkImageActor)

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import sys
import pycaster.pycaster as pycaster
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageQt

from helperFunctions import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(1200, 550)
        self.frame = QtWidgets.QFrame()
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)

        MainWindow.setWindowIcon(QtGui.QIcon('optics.png'))
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
        # self.vtkWidget.GetRenderWindow().AddRenderer(self.ren2)

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        style = vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self.vBoxLayout = QtWidgets.QVBoxLayout()
        self.pushButton = QtWidgets.QPushButton(" Render \n raytracing")
        self.pushButton.setFont(QtGui.QFont('Times', 12))
        self.sliderx = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderx.setValue(50)
        self.slidery = QtWidgets.QSlider(Qt.Horizontal)
        self.slidery.setValue(100)
        self.sliderz = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderz.setValue(50)
        self.sliderCx = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderCx.setValue(50)
        self.sliderCy = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderCy.setValue(50)
        self.sliderCz = QtWidgets.QSlider(Qt.Horizontal)
        self.sliderCz.setValue(50)
        self.textRotation = QtWidgets.QLineEdit("")
        rotationButton = QtWidgets.QPushButton("Rotate")
        rotationButton.setFont(QtGui.QFont('Times', 8))
 

        self.vBoxLayout.addWidget(self.pushButton)
        self.vBoxLayout.addWidget(QtWidgets.QLabel("Move Light"))
        self.vBoxLayout.addWidget(self.sliderx)
        self.vBoxLayout.addWidget(self.slidery)
        self.vBoxLayout.addWidget(self.sliderz)
        self.vBoxLayout.addWidget(QtWidgets.QLabel("Move Camera"))
        self.vBoxLayout.addWidget(self.sliderCx)
        self.vBoxLayout.addWidget(self.sliderCy)
        self.vBoxLayout.addWidget(self.sliderCz)
        self.vBoxLayout.addWidget(QtWidgets.QLabel("Rotate Body"))
        self.vBoxLayout.addWidget(self.textRotation)
        self.vBoxLayout.addWidget(rotationButton)
        self.vBoxLayout.addWidget(QtWidgets.QLabel())
        self.vBoxLayout.setStretch(13, 1)


        self.hBoxLayout = QtWidgets.QHBoxLayout()
        self.hBoxLayout.addWidget(self.vtkWidget)
        # defaultImage = np.zeros((500, 500))
        # width, height = defaultImage.shape
        # bytesPerLine = 3 * width
        # qImg = QtGui.QImage(defaultImage.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # self.labelImage = QtWidgets.QLabel()
        # self.labelImage.setPixmap(QtGui.QPixmap(qImg))
        # self.hBoxLayout.addWidget(self.labelImage)
        self.hBoxLayout.addLayout(self.vBoxLayout)
        self.hBoxLayout.setStretch(0, 1)
        # self.hBoxLayout.setStretch(1, 1)
        # self.labelImage.adjustSize()

        self.pushButton.clicked.connect(self.rayTrancingRender)
        self.sliderx.valueChanged.connect(self.moveAgentx)
        self.slidery.valueChanged.connect(self.moveAgenty)
        self.sliderz.valueChanged.connect(self.moveAgentz)
        self.sliderCx.valueChanged.connect(self.moveCamerax)
        self.sliderCy.valueChanged.connect(self.moveCameray)
        self.sliderCz.valueChanged.connect(self.moveCameraz)
        rotationButton.clicked.connect(self.rotateBody)
        
        self.frame.setLayout(self.hBoxLayout)
        self.setCentralWidget(self.frame)

        self.vtkWidget.GetRenderWindow().Render()
        self.show()
        self.iren.Initialize()
        self.iren.Start()
    
    def createRenderers(self):
        self.ren = vtkRenderer()
        # self.ren2 = vtkRenderer()
        self.mainActors = []
        self.objects = []
        self.actors = []
        self._createCamera()
        self._addActors()

        self.ren.ResetCamera()
        # self.ren.SetViewport(0, 0, 0.5, 1)
        # self.ren2.ResetCamera()
        self.ren.GetActiveCamera().Zoom(0.7)
        # self.ren2.SetViewport(0.5, 0, 1, 1)
    
    def _createCamera(self):
        self.width = 50
        self.height = 50

        self.max_depth = 3

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

    def _addActors(self):
        # Set the background color.
        bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
        self.colors.SetColor("ivory_black", *bkg)

        configs = [ {"image_file": "LogoMines.jpg", "angle": 0},
                    {"image_file": "cloud.jpg", "angle": 90},
                    {"image_file": "hand-7014643_1920.jpg", "angle": 180},
                    {"image_file": "LogoMines.jpg", "angle": 270},
        ]

        for config in configs:
            reader = vtkJPEGReader()
            reader.SetFileName(config["image_file"])
            texture = vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())

            plane = vtkPlane()
            plane.SetOrigin(0.6, 0, 0)
            plane.SetNormal(1, 0, 0)

            cube = vtkCubeSource()
            cube.SetXLength(1)
            cube.SetYLength(1)
            cube.SetZLength(1)
            cube.SetCenter((0.2, 0, 0))

            map_to_plane = vtkTextureMapToPlane()
            map_to_plane.SetInputConnection(cube.GetOutputPort())

            #clipper = vtkClipDataSet()
            clipper = vtkClipPolyData()
            clipper.SetInputConnection(cube.GetOutputPort())
            clipper.GenerateClippedOutputOn()
            clipper.SetClipFunction(plane)
            clipper.SetValue(0)
            clipper.Update()
            # self.objects.append(clipper)

            #image_mapper = vtkDataSetMapper()
            image_mapper = vtkPolyDataMapper()
            image_mapper.SetInputConnection(clipper.GetOutputPort())

            planeActor = vtkActor()
            planeActor.SetMapper(image_mapper)
            planeActor.SetTexture(texture)
            # planeActor.DragableOn()
            # planeActor.SetDragable(1)
            # planeActor.GetProperty().SetColor(self.colors.GetColor3d("White"))
            planeActor.GetProperty().SetColor([0.9, 0.9, 0.9])
            planeActor.GetProperty().SetAmbientColor([0.1, 0.1, 0.1])
            planeActor.GetProperty().SetDiffuseColor([0.9, 0.9, 0.9])
            planeActor.GetProperty().SetSpecularColor([1, 1, 1])

            boundaryEdges = vtkFeatureEdges()
            boundaryEdges.SetInputConnection(clipper.GetOutputPort())
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
            # boundaryActor.GetProperty().SetColor(self.colors.GetColor3d("White"))

            planeActor.RotateY(config["angle"])
            boundaryActor.RotateY(config["angle"])
            # planeActor.GetProperty().SetRepresentation(1)
            # boundaryActor.GetProperty().SetRepresentation(1)
            self.ren.AddActor(planeActor)
            self.ren.AddActor(boundaryActor)
            self.mainActors.append(planeActor)
            self.mainActors.append(boundaryActor)
            # self.actors.append(planeActor)
            

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
        self.mainCubeActor = mainCubeActor
        self.mainActors.append(mainCubeActor)
        self.actors.append(mainCubeActor)
        # mainCubeActor.GetProperty().SetRepresentation(1)

        cubeRef = vtkCubeSource()
        cubeRef.SetXLength(6)
        cubeRef.SetYLength(3.4)
        cubeRef.SetZLength(8)
        cubeRef.SetCenter((0, 1.1, 1.5))
        # self.objects.append(cubeRef)

        outline = vtkOutlineFilter()
        outline.SetInputConnection(cubeRef.GetOutputPort())
        outlineMapper = vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtkActor()
        outlineActor.SetMapper(outlineMapper)

        cubeRefMapper = vtkDataSetMapper()
        cubeRefMapper.SetInputConnection(cubeRef.GetOutputPort())
        cubeRefActor = vtkActor()
        cubeRefActor.SetMapper(cubeRefMapper)
        cubeRefActor.GetProperty().SetColor([0.6, 0.6, 0.6])
        cubeRefActor.GetProperty().SetAmbientColor([0.1, 0.1, 0.1])
        cubeRefActor.GetProperty().SetDiffuseColor([0.6, 0.6, 0.6])
        cubeRefActor.GetProperty().SetSpecularColor([1, 1, 1])
        # self.actors.append(cubeRefActor)

        self._createSunActor()

        self.ren.AddActor(outlineActor)
        self.ren.AddActor(mainCubeActor)
        self.ren.AddActor(planeActor)
        self.ren.AddActor(self.sunActor)
        self.ren.SetBackground(self.colors.GetColor3d("slate_blue_medium")) # "BkgColor"

        # self._createLight()
        # self.ren.AddLight(self.light)

        # self.ren2.SetBackground(self.colors.GetColor3d("mediumseagreen"))

    def _createSunActor(self):
        ResolutionSun = 10
        sun = vtkSphereSource()
        sun.SetCenter(0.0, 2.7, 1.4)
        sun.SetRadius(0.1)
        sun.SetThetaResolution(ResolutionSun)
        sun.SetPhiResolution(ResolutionSun)
        #sun.SetStartTheta(180)  # create a half-sphere
        self.sun = sun
        mapperSun = vtkPolyDataMapper()
        mapperSun.SetInputConnection(sun.GetOutputPort())
        self.sunActor = vtkActor()
        self.sunActor.SetMapper(mapperSun)
        self.sunActor.GetProperty().SetColor([1.0, 1.0, 0.0]) 

    
    def _createLight(self):
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
        
        # print(self.camera, self.actorCamera.GetPosition())
        self.camera = self.initCamera + l2n(self.actorCamera.GetPosition())
        self.screen = coordsScreen(self.screen_size, self.camera, self.ratio)
        self.vtkWidget.GetRenderWindow().Render()
        # print(self.camera)

    
    def rotateBody(self):
        if self.textRotation.text() and self.textRotation.text()!="-":
            angle = float(self.textRotation.text())
            for actor in self.mainActors:
                actor.RotateY(angle)
            self.vtkWidget.GetRenderWindow().Render()

    def nearest_intersected_object(self, objects, origin, direction):
        # after, object will become objects
        for object in objects:
            pTarget = origin + 40*direction
            obb = vtkOBBTree()
            obb.SetDataSet(object.GetOutput())
            obb.BuildLocator()
            distances = []
            cellIds = []
            if isHit(obb, origin, pTarget): 
                pointsInter, cellIdsInter = GetIntersect(obb, origin, pTarget)
                #caster = pycaster.rayCaster(object)
                # pointsIntersection = caster.castRay(origin, pTarget)
                firstPoint = pointsInter[0]
                cellId = cellIdsInter[0]
                distance = np.linalg.norm(l2n(firstPoint) - origin)
                distances.append(distance)
                cellIds.append(cellId)
                # print(firstPoint, distance, cellId)
                # addLine(self.ren, origin, firstPoint)
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
        # self.ren2.SetBackground(self.colors.GetColor3d("alice_blue"))
        bl = 0
        nbl = 0

        image = np.zeros((self.height, self.width, 3))
        for i, y in enumerate(np.linspace(self.screen[1], self.screen[3], self.height)):
            for j, x in enumerate(np.linspace(self.screen[0], self.screen[2], self.width)):
                # screen is on origin
                pixel = np.array([x, y, self.camera[2] - self.distScreen])
                origin = self.camera
                direction = normalize(pixel - origin)
                addLine(self.ren, origin, pixel, color=[0.0, 0.0, 1.0])
                # print("Added", pixel)
                # self.vtkWidget.GetRenderWindow().Render()

                color = np.zeros((3))
                reflection = 1

                for k in range(self.max_depth): #self.max_depth
                    # check for intersections
                    nearest_object, min_distance, cellId = self.nearest_intersected_object(
                                                                self.objects, origin, direction)
                            
                    if nearest_object is None:
                        bl+=1
                        break
                    else:
                        nbl+=1
              
                    intersection = origin + min_distance * direction
                    # addLine(self.ren, origin, intersection, color=[0.0, 0.0, 1.0])
                    normal_to_surface = calcNormals(nearest_object, cellId)
      
                    shifted_point = intersection + 1e-5 * normal_to_surface
                    intersection_to_light = normalize(l2n(self.sunActor.GetCenter()) - shifted_point)
                    # print(min_distance, intersection_to_light_distance)

                    _, min_distance, _ = self.nearest_intersected_object(self.objects, 
                                                            shifted_point, intersection_to_light)

                    intersection_to_light_distance = np.linalg.norm(l2n(self.sunActor.GetCenter()) - intersection)
                    is_shadowed = min_distance < intersection_to_light_distance

                    if is_shadowed:
                        bl+=1
                        break
                    else:
                        nbl+=1
                    
                    illumination = np.zeros((3))
                    ## PAY ATTENTION to nearest Object
                    # nearestObject = self.mainCubeActor
                    index = self.objects.index(nearest_object)
                    actor = self.actors[index]
                    prop = actor.GetProperty()
                    shininess = 100
                    reflection = 0.5

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
                    reflection *= reflection

                    origin = shifted_point
                    direction = reflected(direction, normal_to_surface)

                image[i, j] = np.clip(color, 0, 1)
     
        print("black", bl)
        print("not black", nbl)
        # print(image)
        # vtkImage = updateFrames(image)
        # image_actor = vtkImageActor()
        # image_actor.SetInputData(vtkImage)
        # self.ren2.AddActor(image_actor)
        # print(image.max())
        image = cv2.resize(image, dsize=(320, int(320/self.ratio)), interpolation=cv2.INTER_CUBIC)
        image -= image.min()
        image /= image.max()
        # print(image.max(), image.min())
        # height, width, channel = image.shape
        # bytesPerLine = 3 * width + 1
        # qImg = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # img = Image.fromarray(image, mode='RGB')
        # qImg = ImageQt.ImageQt(img)
        # labelImage = QtWidgets.QLabel()
        # self.labelImage.setPixmap(QtGui.QPixmap.fromImage(qImg))
        # self.labelImage.adjustSize()
        plt.imsave('RayTracing.png', image)
        # plt.imshow(image)
        # self.vtkWidget.GetRenderWindow().AddRenderer(self.ren2)
        # self.vtkWidget.GetRenderWindow().Render()
        # window.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())