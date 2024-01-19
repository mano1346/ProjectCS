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



class TimeSliderCallback:
    def __init__(self, satellite_coords, actors):
        self.satellite_coords = satellite_coords
        self.actors = actors
        self.current_time = -1
    
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



def visualize_data(satellite_coords):
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


    slider_widget = make_slider_widget(len(satellite_coords))
    slider_widget.SetInteractor(iren)

    callback = TimeSliderCallback(satellite_coords, actors)
    callback.update_actors(0)
    
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
        ])