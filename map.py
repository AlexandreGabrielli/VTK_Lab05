# coding=utf-8
import vtk
import math
import pyproj
import numpy as np
import datetime as dt


def convertRT90_to_WGS84(y,x):
    """
    :param x: coordinate x in meters
    :param y: coordinate y in meters
    :return: longitude and latitue
    """
    transformer = pyproj.Transformer.from_crs("epsg:3021", "epsg:4326")
    return transformer.transform(x, y)


def convertWGS84_to_RT90(longitude, latitude):
    """

    :param longitude: in degres
    :param latitude: in degres
    :return: coordinate x , coordinate y
    """
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3021")
    return transformer.transform(longitude, latitude)


# couleur rgb bleu vtk
blue = [0.19, 0.55, 0.9]
'''
Haut-gauche: 1349340 7022573
Haut-droite: 1371573 7022967
Bas-droite: 1371835 7006362
Bas-gauche: 1349602 7005969 
'''
# The Map RT90 coordinates (Given data)
RT90_MAP_TOP_LEFT = (1349340, 7022573)
RT90_MAP_TOP_RIGHT = (1371573, 7022967)
RT90_MAP_BOTTOM_RIGHT = (1371835, 7006362)
RT90_MAP_BOTTOM_LEFT = (1349602, 7005969)

RG84_MAP_TOP_LEFT = convertRT90_to_WGS84(RT90_MAP_TOP_LEFT[0], RT90_MAP_TOP_LEFT[1])
RG84_MAP_TOP_RIGHT = convertRT90_to_WGS84(RT90_MAP_TOP_RIGHT[0], RT90_MAP_TOP_RIGHT[1])
RG84_MAP_BOTTOM_RIGHT = convertRT90_to_WGS84(RT90_MAP_BOTTOM_RIGHT[0], RT90_MAP_BOTTOM_RIGHT[1])
RG84_MAP_BOTTOM_LEFT = convertRT90_to_WGS84(RT90_MAP_BOTTOM_LEFT[0], RT90_MAP_BOTTOM_LEFT[1])

latitude_min = min(RG84_MAP_TOP_LEFT[1], RG84_MAP_BOTTOM_LEFT[1])
lattitude_max = max(RG84_MAP_TOP_RIGHT[1], RG84_MAP_BOTTOM_RIGHT[1])
longitude_min = min(RG84_MAP_BOTTOM_LEFT[0], RG84_MAP_BOTTOM_RIGHT[0])
longitude_max = max(RG84_MAP_TOP_LEFT[0], RG84_MAP_TOP_RIGHT[0])


def transforme(latitude, longitude, altitude):
    """
    cette fonction calculer les cohordonner (x,y,z) d'un point en partant du principe
    que la terre est parfaitement ronde et que le centre de la terre se situe en (0,0,0)

    :param latitude: la latitude du point (en radian)
    :param longitude: la longitude du point (en radian)
    :param altitude:  l'altitude par rapport à la croute terrestre (en metre)
    :return: coordonne corriger du point (x,y,z) (en mettre)
    """
    # le points se trouve sur une sphére egal au rayon de la terre
    # plus son altitude par rapport au niveau de la mer.
    rayon = 6371009 + altitude
    x = rayon * math.cos(latitude) * math.cos(longitude)
    y = rayon * math.cos(latitude) * math.sin(longitude)
    z = rayon * math.sin(latitude)
    # print("x : {:f} , y {:f} , z {:f}".format(x,y,z))
    return x, y, z


def computeVerticalSpeed(gliderTrajectory, index):
    """ Calculate the verical speed of the glider given its trajectory
    and a position index.
    Args:
        gliderTrajectory - Array of lists representing the glider position
        at different times.
        index - position in the array at which to compute the vertical speed.
        Must be > 1
    Returns:
        The vertical speed value
    """
    lastDateTime = gliderTrajectory[index - 1][3]
    dateTime = gliderTrajectory[index][3]
    lastAltitude = gliderTrajectory[index - 1][2]
    altitude = gliderTrajectory[index][2]
    return (altitude - lastAltitude) / (dateTime - lastDateTime).total_seconds()


def MakeLUT():
    """
    Make a lookup table using vtkColorSeries.

    :return: : An indexed lookup table.
    """

    # Make the lookup table.
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.5, 0)
    lut.SetSaturationRange(0.5, 0)
    lut.SetValueRange(0.5, 1)

    # for water
    lut.SetBelowRangeColor(blue[0], blue[1], blue[2], 1)
    lut.UseBelowRangeColorOn()
    lut.Build()
    return lut


def mapCoordinatesToTexture(lat, longitude):
    px = [RG84_MAP_BOTTOM_LEFT[1], RG84_MAP_BOTTOM_RIGHT[1], RG84_MAP_TOP_RIGHT[1], RG84_MAP_TOP_LEFT[1]]
    py = [RG84_MAP_BOTTOM_LEFT[0], RG84_MAP_BOTTOM_RIGHT[0], RG84_MAP_TOP_RIGHT[0], RG84_MAP_TOP_LEFT[0]]

    coeff = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 1, 0]
    ])
    coeff_inv = np.linalg.inv(coeff)
    a = np.dot(coeff_inv, px)
    b = np.dot(coeff_inv, py)

    # quadratic equation coeffs, aa*mm^2+bb*m+cc=0
    aa = a[3] * b[2] - a[2] * b[3]
    bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + longitude * b[3] - lat * a[3]
    cc = a[1] * b[0] - a[0] * b[1] + longitude * b[1] - lat * a[1]

    # compute m = (-b+sqrt(b^2-4ac))/(2a)
    det = math.sqrt(bb * bb - 4 * aa * cc)
    m = (-bb + det) / (2 * aa)

    # compute l
    l = (longitude - a[0] - a[2] * m) / (a[1] + a[3] * m)
    return l, m


def make_map_points():
    # The min and max on x and y axis of the Map

    print(longitude_min)
    print(longitude_max)
    print(latitude_min)
    print(lattitude_max)

    delta_degre = 5 / 6000

    top_index = int((65 - longitude_max) / delta_degre)
    bottom_index = int((65 - longitude_min) / delta_degre)
    left_index = int((latitude_min - 10) / delta_degre)
    right_index = int((lattitude_max - 10) / delta_degre)

    data_map = np.fromfile("data/EarthEnv-DEM90_N60E010.bil", dtype=np.int16).reshape(6000, 6000)
    print(top_index)
    print(bottom_index + 1)
    print(left_index)
    print(right_index)

    data_map = data_map[top_index:bottom_index + 1, left_index:right_index + 1]
    points = vtk.vtkPoints()
    scalar = vtk.vtkFloatArray()
    # Texture coordinates values
    textureCoordinates = vtk.vtkFloatArray()
    textureCoordinates.SetNumberOfComponents(2)

    for i, row in enumerate(data_map):
        for j, altitude in enumerate(row):
            lattitude = longitude_max - i * delta_degre
            longitude = latitude_min + j * delta_degre
            point_x, point_y, point_Z = transforme(math.radians(lattitude), math.radians(longitude), altitude)
            points.InsertNextPoint(point_x, point_y, point_Z)
            scalar.InsertNextValue(altitude)
            cx, cy = mapCoordinatesToTexture(lattitude, longitude)
            textureCoordinates.InsertNextTuple((cx, cy))

    return points, scalar, data_map.shape, textureCoordinates


points, scalar, dimensions, textureCoordinates = make_map_points()

# nous avons choisi une structured grid car cette structure permet de réaliser une topologie réguliere de point.
struGrid = vtk.vtkStructuredGrid()
struGrid.SetPoints(points)
struGrid.SetDimensions(dimensions[0],dimensions[1], 1)
struGrid.GetPointData().SetTCoords(textureCoordinates)

# Load texture from JPEG
JPEGReader = vtk.vtkJPEGReader()
JPEGReader.SetFileName("data/glider_map.jpg")
texture = vtk.vtkTexture()
texture.SetInputConnection(JPEGReader.GetOutputPort())

# mapper
gridMapper = vtk.vtkDataSetMapper()
gridMapper.SetInputData(struGrid)

# actor
gridActor = vtk.vtkActor()
gridActor.SetMapper(gridMapper)
gridActor.SetTexture(texture)
gridActor.GetProperty().SetPointSize(3)

# glider
print("glider")
vtkPoints = vtk.vtkPoints()
speedScalar = vtk.vtkFloatArray()
gliderTrajectory = []
minVerticalSpeed = 0
maxVerticalSpeed = 0
lineCount = -1
with open("data/vtkgps.txt") as fileIn:
    for line in fileIn:
        if lineCount == -1:
            lineCount += 1
            continue
        lineCount += 1

        # Get values of a line as an array of String
        values = line.split()
        # Read position and time coordinates
        coordinates = [int(values[1]), int(values[2]), float(values[3])]

        dateArray = values[4].split('/')
        timeArray = values[5].split(':')
        dateTime = dt.datetime(int(dateArray[0]), int(dateArray[2]), int(dateArray[1]), int(timeArray[0]),
                               int(timeArray[1]), int(timeArray[2]))
        coordinates.append(dateTime)
        gliderTrajectory.append(coordinates)
        longlat = convertRT90_to_WGS84(int(values[1]), int(values[2]))

        # make reall word coordinate
        p_x, p_y, p_z = transforme(math.radians(longlat[1]), math.radians(longlat[0]), coordinates[2])
        vtkPoints.InsertNextPoint((p_x, p_y, p_z))
        i = 0
        if (lineCount - 1) > 0:
            verticalSpeed = computeVerticalSpeed(gliderTrajectory, lineCount - 1)
            if verticalSpeed < minVerticalSpeed:
                minVerticalSpeed = verticalSpeed
            if verticalSpeed > maxVerticalSpeed:
                maxVerticalSpeed = verticalSpeed
            speedScalar.InsertTuple1(i, verticalSpeed)
            i += 1

polyLine = vtk.vtkPolyLine()
polyLine.GetPointIds().SetNumberOfIds(vtkPoints.GetNumberOfPoints())
for i in range(0, vtkPoints.GetNumberOfPoints()):
    polyLine.GetPointIds().SetId(i, i)
cells = vtk.vtkCellArray()
cells.InsertNextCell(polyLine)

polyData = vtk.vtkPolyData()
polyData.SetPoints(vtkPoints)
polyData.SetLines(cells)
polyData.GetPointData().SetScalars(speedScalar)

lut = MakeLUT()
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(polyData)
mapper.SetLookupTable(lut)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a tube (cylinder) around the line
tubeFilter = vtk.vtkTubeFilter()
tubeFilter.SetInputData(polyData)
tubeFilter.SetRadius(20)
tubeFilter.SetNumberOfSides(50)
tubeFilter.Update()

# Create a mapper and actor
tubeMapper = vtk.vtkPolyDataMapper()
tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())
tubeActor = vtk.vtkActor()
tubeActor.SetMapper(tubeMapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(gridActor)
renderer.AddActor(tubeActor)
renderer.AddActor(actor)
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)

renWin.SetSize(800, 800)

ren = vtk.vtkRenderWindowInteractor()
ren.SetRenderWindow(renWin)
style = vtk.vtkInteractorStyleTrackballCamera()

ren.SetInteractorStyle(style)
ren.Initialize()
ren.Start()
