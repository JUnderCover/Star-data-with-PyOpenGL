# PyOpenGL

This app renders star data using OpenGL and has basic naviation features. All rendering is done via the Modern OpenGL approach with custom defined vertex and fragment shaders. All matrix calculations are done via numpy and numpy arrays are passed directly into OpenGL functions.

## Controls

* _Mouse Movement_: Rotates the entire scene and moves cursor
* _R_: Toggle rotate mode on and off. When off, one can move the cursor without rotating the scene. 
* _O_: Switch camera focus to origin
* _Q_: Quit and close window
* _Press Mouse Wheel and Scroll_: Zooming in and out. Scrolling the mouse wheel alone will do nothing.
* _Arrow Keys_: Translate camera foward, backward, left or right.
* _Left Click (when cursor is diamond)_: change camera focus to selected point
* _Right Click_: Reset to initial view


### Installing

See requirements.txt or PipFile for the required packages. If shader compilation fails due to incorrect GLSL version, try the following. 


At the top of **both** vertexshader.glsl and fragmentshader.glsl, change:

```
#version 440
```

to something like

```
#version 130
```

...or which ever GLSL version is support on your machine. 

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds



## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Very useful Modern OpenGL tutorials by Joey de Vries: https://learnopengl.com/

Matrix methods derived from: https://github.com/mackst/glm/tree/master/glm

Camera translation and rotation explaination: https://gamedev.stackexchange.com/questions/20758/how-can-i-orbit-a-camera-about-its-target-point

Mouse selection explaination: https://www.bfilipek.com/2012/06/select-mouse-opengl.html

