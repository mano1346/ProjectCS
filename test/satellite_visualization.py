from __future__ import annotations

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
    sphere = vtkSphereSource()
    sphere.SetRadius(50.0)
    sphere.SetPhiResolution(10)
    sphere.SetThetaResolution(10)
    
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    colors = vtkNamedColors()
    renderer = vtkRenderer()

    actors = []
    for _ in range(len(satellite_coords[0])):
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d('White'))

        renderer.AddActor(actor)
        actors.append(actor)
    

    earth = vtkSphereSource()
    earth.SetRadius(6378)
    earth.SetPhiResolution(200)
    earth.SetThetaResolution(200)

    earthMapper = vtkPolyDataMapper()
    earthMapper.SetInputConnection(earth.GetOutputPort())

    earthActor = vtkActor()
    earthActor.SetMapper(earthMapper)
    earthActor.GetProperty().SetColor(colors.GetColor3d('CadetBlue'))
    renderer.AddActor(earthActor)


    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(960, 540)
    renderWindow.SetWindowName('Satellite visualization')

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renderWindow)
    iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

    chartUpdater = None
    if (hist_counts != None):
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



def make_slider_widget(length):
    slider = vtkSliderRepresentation2D()

    slider.SetMinimumValue(0)
    slider.SetMaximumValue(length - 1.0)
    slider.SetValue(0)

    slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint1Coordinate().SetValue(0.1, 0.05)
    slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint2Coordinate().SetValue(0.9, 0.05)

    slider.SetTubeWidth(0.005)
    slider.SetSliderLength(0.02)
    slider.SetSliderWidth(0.02)
    slider.SetEndCapLength(0.0)

    slider_widget = vtkSliderWidget()
    slider_widget.SetRepresentation(slider)

    return slider_widget



if __name__ == '__main__':
    visualize_data([
        [[5152,5124,-5331], [-5200,-5153,5430]],
        [[5192,5124,-4331], [-5150,-5153,4430]]
        ], 
        [[2, 5, 10, 20, 25], [0, 4, 9, 22, 31]],
        [10, 20, 30, 40, 50])