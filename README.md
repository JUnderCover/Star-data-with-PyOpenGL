# PyOpenGL

This app renders star data using OpenGL and has basic naviation features. All rendering is done via the Modern OpenGL approach with custom defined vertex and fragment shaders. All matrix calculations are done via numpy and numpy arrays are passed directly into OpenGL functions.

All data is in p2data_v2.txt. Data was provided by Dr. John Wise of Georgia Tech.  

### Controls

* _Mouse Movement_: Rotates the entire scene and moves cursor.
* _R_: Toggle rotate mode on and off. When off, one can move the cursor without rotating the scene.
* _O_: Switch camera focus to origin.
* _Q_: Quit and close window.
* _Press Mouse Wheel and Scroll_: Zooming in and out. Scrolling the mouse wheel alone will do nothing.
* _Arrow Keys_: Translate camera foward, backward, left or right.
* _Left Click (when cursor is diamond)_: change camera focus to selected point.
* _Right Click_: Reset to initial view.


### Installing

See requirements.txt or PipFile for the required packages. Note that the installation of yt is optional (consider removing it from PipFile). Code was developed with Python 3.6, but should be compatible with Python 2. 

To create scene run:

```
python render.py
```

If shader compilation fails due to incorrect GLSL version, try the following.


At the top of **both** vertexshader.glsl and fragmentshader.glsl, change:

```
#version 440
```

to something like

```
#version 130
```

...or which ever GLSL version is support on your machine.


### Issues

Pygame and PyOpenGL have certain compatability issues. Pygame blit will not work. See https://stackoverflow.com/questions/40207529/blitting-pygame-surface-onto-pygame-opengl-display. This makes displaying text and other basic shapes with Pygame rather non-trivial.


Running render.py might produce the following terminal message:

```
Unable to load numpy_formathandler accelerator from OpenGL_accelerate
```

However, there should be no noticeable rendering or performance issues. See https://stackoverflow.com/questions/20678260/pyopengl-accelerate-numpy, which reccomends to install PyOpenGL using easy_install instead of within a virutal enviornment. 



## Acknowledgments

Very useful Modern OpenGL tutorials by Joey de Vries: https://learnopengl.com/

Matrix methods derived from: https://github.com/mackst/glm/tree/master/glm

Camera translation and rotation explaination: https://gamedev.stackexchange.com/questions/20758/how-can-i-orbit-a-camera-about-its-target-point

Mouse selection explaination: https://www.bfilipek.com/2012/06/select-mouse-opengl.html

_Big thanks to Dr. Wise for his time and guidance!_
