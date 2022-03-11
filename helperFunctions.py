from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtk import vtkPolyDataNormals, vtkOBBTree, vtkIdList, vtkLineSource, VTK_UNSIGNED_CHAR             
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
from vtkmodules.vtkFiltersSources import vtkSphereSource

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt5.QtGui import QIcon, QPixmap

import numpy as np


l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

def coordsScreen(screen_size, camera, ratio):
    screen = (-screen_size + camera[0], screen_size / ratio + camera[1],
              screen_size + camera[0], -screen_size / ratio + camera[1])
    return screen

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def isHit(obbTree, pSource, pTarget):
    r"""Returns True if the line intersects with the mesh in 'obbTree'"""
    code = obbTree.IntersectWithLine(pSource, pTarget, None, None)
    if code==0:
        return False
    return True
    
def GetIntersect(obbTree, pSource, pTarget):
    points = vtkPoints()
 
    cellIds = vtkIdList()
    code = obbTree.IntersectWithLine(pSource, pTarget, points, cellIds)
    pointData = points.GetData()
    noPoints = pointData.GetNumberOfTuples()
    noIds = cellIds.GetNumberOfIds()
    
    assert (noPoints == noIds)
    
    pointsInter = []
    cellIdsInter = []
    for idx in range(noPoints):
        pointsInter.append(pointData.GetTuple3(idx))
        cellIdsInter.append(cellIds.GetId(idx))
    
    return pointsInter, cellIdsInter

def getObbs(objects):
    obbs = []
    for object in objects:
        obb = vtkOBBTree()
        obb.SetDataSet(object.GetOutput())
        obb.BuildLocator()
        obbs.append(obb)
    return obbs

def calcNormals(object, cellId, direction):
    normalsCalc = vtkPolyDataNormals()
    normalsCalc.SetInputConnection(object.GetOutputPort())
    normalsCalc.ComputePointNormalsOff()
    normalsCalc.ComputeCellNormalsOn()
    normalsCalc.SplittingOff()
    normalsCalc.Update()
    normalsObject = normalsCalc.GetOutput().GetCellData().GetNormals()
    normal = l2n(normalsObject.GetTuple(cellId))
    
    # Get the right normal
    cos = np.dot(normal, direction) / np.linalg.norm(normal) /  np.linalg.norm(direction)
    if cos > 0:
        normal = -normal

    return normal

def addLine(renderer, p1, p2, color=[0.0, 0.0, 1.0]):
    "Function used only for debugging"
    line = vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    renderer.AddActor(actor)

def addPoint(renderer, p, radius=0.01, color=[0.0, 0.0, 0.0]):
    "Function used only for debugging"
    point = vtkSphereSource()
    point.SetCenter(p)
    point.SetRadius(radius)
    point.SetPhiResolution(100)
    point.SetThetaResolution(100)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(point.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    renderer.AddActor(actor)

def showImage():
        win = QDialog()
        win.setWindowTitle("Raytracing Image")
        win.setWindowIcon(QIcon('optics.png'))

        imageLabel = QLabel()
        imageLabel.setPixmap(QPixmap("Raytracing.png"))
        layout = QVBoxLayout()
        layout.addWidget(imageLabel)

        win.setLayout(layout)
        win.exec_()

