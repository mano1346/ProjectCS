"""
University of Amsterdam

Course: Project Computational Science
Authors: Emmanuel Mukeh, Justin Wong & Arjan van Staveren

This code creates a 3d visualisation based on data gathered in the simulation.
It is used in final_model.py.
"""

import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderer,
    vtkRenderWindowInteractor,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import (
    vtkSliderRepresentation2D,
    vtkSliderWidget
)
from vtkmodules.all import (
    vtkContextView,
    vtkTable,
    vtkChartXY,
    vtkChart,
    vtkIntArray
)


# The __call__ function should be called when the slider changes position.
# This class will then update the positions of the actors to match the positions of the satellites at the selected time.
# It also prompts a ChartUpdater to update the shown histogram.
class TimeSliderCallback:
    def __init__(self, satellite_coords, actors, chartUpdater):
        self.satellite_coords = satellite_coords
        self.actors = actors
        self.current_time = -1
        self.chartUpdater = chartUpdater
    
    def __call__(self, caller, ev):
        new_time = round(caller.GetRepresentation().GetValue())
        self.update_actors(new_time)
    
    def update_actors(self, time):
        if time == self.current_time:
            return
        self.current_time = time
        
        coordinates = self.satellite_coords[time]
        for i, actor in enumerate(self.actors):
            actor.SetPosition(coordinates[i])
        
        if (self.chartUpdater != None):
            self.chartUpdater.update_chart(time)


# This will update the histogram based on the given time.
class ChartUpdater:
    def __init__(self, table, bar, renderWindow, chart):
        self.table = table
        self.bar = bar
        self.renderWindow = renderWindow
        self.chart = chart
    
    def update_chart(self, time):
        self.bar.SetInputData(self.table, 0, time + 1)
        self.chart.RecalculateBounds()
        self.renderWindow.Render()


def visualize_data(satellite_coords, hist_counts = None, bins = None):
    # The sphere model of a satellite is defined.
    sphere = vtkSphereSource()
    sphere.SetRadius(50.0)
    sphere.SetPhiResolution(10)
    sphere.SetThetaResolution(10)
    
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    colors = vtkNamedColors()
    renderer = vtkRenderer()

    # An actor is added to the renderer for each satellite, which all reuse the same mapper.
    actors = []
    for _ in range(len(satellite_coords[0])):
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d('White'))

        renderer.AddActor(actor)
        actors.append(actor)
    
    # The sphere model of Earth is defined (larger and higher resolution).
    earth = vtkSphereSource()
    earth.SetRadius(6378)
    earth.SetPhiResolution(200)
    earth.SetThetaResolution(200)

    earthMapper = vtkPolyDataMapper()
    earthMapper.SetInputConnection(earth.GetOutputPort())

    # The Earth is added as an actor to the renderer.
    earthActor = vtkActor()
    earthActor.SetMapper(earthMapper)
    earthActor.GetProperty().SetColor(colors.GetColor3d('CadetBlue'))
    renderer.AddActor(earthActor)


    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(960, 540)
    renderWindow.SetWindowName('Satellite visualization')

    # This interactor makes it possible to move the camera using the mouse.
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)
    iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

    chartUpdater = None
    if (hist_counts != None):
        # The table containing all distance data and its associated window and ChartUpdater are created.
        view = vtkContextView()
        view.GetRenderWindow().SetWindowName('Distance Histogram')

        table = vtkTable()
        vtk_array = vtkIntArray()
        vtk_array.SetName('bins')
        table.AddColumn(vtk_array)

        for i in range(len(hist_counts)):
            vtk_array = vtkIntArray()
            vtk_array.SetName(f'{i}')
            table.AddColumn(vtk_array)
        table.SetNumberOfRows(len(hist_counts[0]))

        for i, bin in enumerate(bins):
            table.SetValue(i, 0, bin)

        for y, col in enumerate(hist_counts):
            for x, bin_value in enumerate(col):
                table.SetValue(x, y + 1, bin_value)

        # The histogram is defined.
        chart = vtkChartXY()
        view.GetScene().AddItem(chart)
        bar = chart.AddPlot(vtkChart.BAR)
        bar.SetColor(150, 150, 150, 255)
        bar.SetInputData(table, 0, 1)

        chartUpdater = ChartUpdater(table, bar, view.GetRenderWindow(), chart)


    callback = TimeSliderCallback(satellite_coords, actors, chartUpdater)
    callback.update_actors(0)
    
    slider_widget = make_slider_widget(len(satellite_coords))
    slider_widget.SetInteractor(iren)

    slider_widget.AddObserver('InteractionEvent', callback)
    slider_widget.SetInteractor(iren)
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.EnabledOn()

    iren.Initialize()
    iren.Start()


# Initializes a slider with a bunch of parameters that dictate its appearance.
def make_slider_widget(length):
    slider = vtkSliderRepresentation2D()

    slider.SetMinimumValue(0)
    slider.SetMaximumValue(length - 1.0)
    slider.SetValue(0)

    slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint1Coordinate().SetValue(0.1, 0.05)
    slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint2Coordinate().SetValue(0.9, 0.05)

    slider.SetTubeWidth(0.001)
    slider.SetSliderLength(0.025)
    slider.SetSliderWidth(0.025)
    slider.SetEndCapLength(0.02)

    slider_widget = vtkSliderWidget()
    slider_widget.SetRepresentation(slider)
    slider_widget.SetNumberOfAnimationSteps(length - 1)

    return slider_widget