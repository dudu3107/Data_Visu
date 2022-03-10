from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData

from vtk import vtkPolyDataNormals, vtkOBBTree, vtkIdList, vtkLineSource, VTK_UNSIGNED_CHAR             
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

import numpy as np


l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

def updateFrames(depthFrame):
   #Build vtkImageData here from the given numpy uint8_t arrays.
   ImageData = vtkImageData()
   depthArray = numpy_support.numpy_to_vtk(depthFrame.ravel(), deep=True, array_type=VTK_UNSIGNED_CHAR) 
   # .transpose(2, 0, 1) may be required depending on numpy array order see - https://github.com/quentan/Test_ImageData/blob/master/TestImageData.py

   ImageData.SetDimensions(depthFrame.shape)
  #assume 0,0 origin and 1,1 spacing.
   ImageData.SetSpacing(1, 1, 0)
   ImageData.SetOrigin(0, 0, 0)
   ImageData.GetPointData().SetScalars(depthArray)
   return ImageData

def coordsScreen(screen_size, camera, ratio):
    screen = (-screen_size + camera[0], screen_size / ratio + camera[1],
              screen_size + camera[0], -screen_size / ratio + camera[1])
    # print(camera, screen)
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

def nearest_intersected_object(objects, origin, direction):
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

def calcNormals(object, cellId):
    normalsCalc = vtkPolyDataNormals()
    normalsCalc.SetInputConnection(object.GetOutputPort())
    normalsCalc.ComputePointNormalsOff()
    normalsCalc.ComputeCellNormalsOn()
    normalsCalc.SplittingOff()
    normalsCalc.FlipNormalsOff()
    normalsCalc.AutoOrientNormalsOn()
    normalsCalc.Update()
    normalsObject = normalsCalc.GetOutput().GetCellData().GetNormals()
    # print(normalsObject.GetNumberOfTuples(), cellId)
    normal = l2n(normalsObject.GetTuple(cellId))

    return normal

def addLine(renderer, p1, p2, color=[0.0, 0.0, 1.0]):
    line = vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    renderer.AddActor(actor)
