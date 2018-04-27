import pygame
import random
import numpy
import math
import sys
from ctypes import *
from PIL import Image

from glm.gtc.matrix_transform import *
from glm.detail.type_vec4 import *
from glm.detail.type_mat4x4 import *

import OpenGL.GL.shaders
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def convert(matrix):
    vecList = matrix._Mat4x4__value
    trans = []
    for vec in vecList:
        trans.append(vec.x)
        trans.append(vec.y)
        trans.append(vec.z)
        trans.append(vec.w)
    return trans

def convertXY(x, y, ballradius):
    d = x * x + y * y
    radiusSquared = ballradius * ballradius
    if (d > radiusSquared):
        return Vec3(x, y, 0.0)
    else:
        return Vec3(x, y, math.sqrt(radiusSquared - d))



def main():
    positions = []
    myfile = open("p2data.txt", "r")
    for line in myfile:
         positions.append((float(line[2:12]) - 0.5) * 2)
         positions.append((float(line[14:24]) - 0.5) * 2)
         positions.append((float(line[26:36]) - 0.5) * 2)
    positions = numpy.array(positions, dtype = numpy.float64)
    #N = positions.size/3
    #pos = positions.reshape(N,3)
    #zsort = numpy.argsort(pos[:,2])
    #newpositions = pos[zsort].reshape(N*3)
    #import pdb; pdb.set_trace()
    pygame.init()
    display = (800, 600)
    aspect = display[0] / display [1]
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    myvao = glGenVertexArrays(1)
    glBindVertexArray(myvao)
    myvbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, myvbo)
    glBufferData(GL_ARRAY_BUFFER, sys.getsizeof(positions), positions, GL_STATIC_DRAW)

    
    vertexShaderSource  = """
    #version 130
    in vec3 position;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    uniform vec2 screenSize;
    uniform float spriteSize;


    void main()
    {
         gl_Position = projection * view * model * vec4(position, 1.0);
    }
    """
    fragmentShaderSource = """
    #version 130
    out vec4 FragColor;
    uniform sampler2D tex;
    void main()
    {
         vec4 texColor = texture(tex, gl_PointCoord); 
         //texColor.a = 1.0 - texColor.r;
         FragColor = texColor;
    }
    """
    shader = shaders.compileProgram(shaders.compileShader(vertexShaderSource, GL_VERTEX_SHADER), shaders.compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER))
    positionAttrib = glGetAttribLocation(shader, "position")
    #ctypes.c_void_p(offset)
    floatsize = 4 # 4 bytes, 32 bits
    stride = 3 * floatsize
    glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(positionAttrib)
    numpos = len(positions) / 3
    numpos = int(numpos)



    glEnable(GL_POINT_SPRITE)
    #glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
    
    #texture
    textureobject = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureobject)
    #glEnable(GL_POINT_SPRITE)
    #glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_UPPER_LEFT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    image = Image.open("halo.jpg")
    width, height = image.size
    print (width, height)
    img_data = numpy.array(list(image.getdata()), numpy.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    #Rotation Around Cluster Vectors
    #cameraPos = Vec3(-0.739, -0.637, 0.7)
    #cameraTarget = Vec3(-0.739, -0.637, 0.638)
    
    

    deltaTime = 0.0
    lastFrame = 0.0
    
    initial = True
    yaw = -90.0
    pitch = 0.0
    xoffset = 0.0
    yoffset = 0.0
    lastX = display[0] /  2
    lastY = display[1] / 2
    fov = 45.0

    #Look at Vectors
    worldUp = Vec3(0.0, 1.0, 0.0)
    origin = Vec3(0.0, 0.0, 0.0)
    cameraPos = Vec3(0.0, 0.0, 5.0)
    frontX = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
    frontY = math.sin(math.radians(pitch))
    frontZ = math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
    cameraFront = Vec3(frontX, frontY, frontZ)
    cameraFront = normalizeVec(cameraFront)
    right = cross(cameraFront, worldUp)
    up = cross(right, cameraFront) #normalized internally    
    delta = 0.0
    fovdelta = 0.0
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(True)
    middle = False

    #glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_COLOR)
    #glBlendFunc(GL_SRC_ALPHA, GL_ZERO)

    click = False
    moving = False
    ballradius = min(display[0] / 2, display[1] / 2)
    initial = True
    model = Mat4x4()
    newmodel = Mat4x4()
    oldmodel = Mat4x4()

    xoffset = 0.0
    yoffset = 0.0
    
    while True:
        timeValue = pygame.time.get_ticks()
        deltaTime = timeValue - lastFrame
        lastFrame = timeValue
        cameraSpeed = 0.009 * deltaTime
        #cameraSpeed = 0.2
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEMOTION and click and not moving:
                print ("Start")
                moving = True
                xpos, ypos = pygame.mouse.get_pos()
                xpos = (xpos - (display[0] / 2))
                ypos = ((display[1] / 2) - ypos)
                startRotationVector = convertXY(xpos, ypos, ballradius)
                startRotationVector = normalize(startRotationVector)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: #Left Click Down
                    #Enable Rotate Mode
                    print ("Left Click down")
                    click = True
                if event.button == 4: # ZOOM IN
                    if middle:
                        print ("ZOOM IN")
                        fovdelta = -.2
                if event.button == 5: # ZOOM OUT
                    if middle:
                        print ("ZOOM OUT")
                        fovdelta = 0.2
                if event.button == 2:
                    middle = True
                if event.button == 3: #Right Click to exit
                    pygame.quit()
                    quit()
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button ==1: #Left Click Up
                    print ("STOP")
                    click = False
                    moving = False
                    initial = True
                if event.button == 2:
                    middle = False
                    print ("DONE")
                    fovdelta = 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    delta = right * cameraSpeed * -1
                    #delta = cross(cameraFront, up) * cameraSpeed * -1
                    #cameraPos -= cross(cameraFront, up) * cameraSpeed
                if event.key == pygame.K_RIGHT:
                    delta = right * cameraSpeed
                    #delta = cross(cameraFront, up) * cameraSpeed
                    #cameraPos += cross(cameraFront, up) * cameraSpeed
                if event.key == pygame.K_UP:
                    delta = cameraSpeed * cameraFront
                    #cameraPos += cameraSpeed * cameraFront
                if event.key == pygame.K_DOWN:
                    delta = cameraSpeed * cameraFront * -1
                    #cameraPos -= cameraSpeed * cameraFront
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    delta = Vec3(0.0, 0.0, 0.0)
                if event.key == pygame.K_RIGHT:
                    delta = Vec3(0.0, 0.0, 0.0)
                if event.key == pygame.K_UP:
                    delta = Vec3(0.0, 0.0, 0.0)
                if event.key == pygame.K_DOWN:
                    delta = Vec3(0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        #glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)
        glBindTexture(GL_TEXTURE_2D, textureobject)
        if moving:
            xoffset = 0.0
            yoffset = 0.0
            #update Rotation Vector
            xpos, ypos = pygame.mouse.get_pos()
            xpos = (xpos - (display[0] / 2))
            ypos = ((display[1] / 2) - ypos)
            currentRotationVector = convertXY(xpos, ypos, ballradius)
            currentRotationVector = normalize(currentRotationVector)
            diff = currentRotationVector - startRotationVector
            mag = math.sqrt((diff.x) * (diff.x) + (diff.y) * (diff.y) + (diff.z) * (diff.z))
            if mag > 1E-6:
                rotationAxis = cross(currentRotationVector, startRotationVector) #normalized
                val = dotVec(currentRotationVector, startRotationVector)
                if val > (1 - 1E-10):
                    val = 1.0
                rotationAngle = math.degrees(math.acos(val))
                axis = Vec3(-rotationAxis.x, -rotationAxis.y, -rotationAxis.z)
                newmodel = Mat4x4()
                newmodel = rotate(newmodel, rotationAngle * 2, axis)
        else:
                oldmodel = model
                newmodel = Mat4x4()
        model = newmodel.__mul__(oldmodel)
        modelList = convert(model)
        modelLoc = glGetUniformLocation(shader, "model")
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, modelList)
        view = Mat4x4()
        cameraPos += delta
        #view = translate(view, Vec4(0.0, 0.0, -3.0, 0.0))
        #view = translate(view, Vec4(0.739, 0.637, -0.7, 0.0))
        
        #Rotation Around Cluster
        #radius = 0.2
        #camX = -0.739 + math.sin(timeValue) * radius
        #camZ = 0.638  + math.cos(timeValue) * radius
        #cameraPos = Vec3(camX, -0.637, camZ)
        #view = lookAt(cameraPos, cameraTarget, up)
        
        if not moving:
            xpos, ypos = pygame.mouse.get_pos()
            if xpos > display[0] -5 or ypos > display[1] -5 or xpos < 5 or ypos < 5:
                pygame.mouse.set_pos(display[0] / 2, display[1] / 2)
                xoffset = 0
                yoffset = 0
                lastX = display[0] / 2
                lastY = display[1] / 2
            if initial:
                pygame.mouse.set_pos(display[0] / 2, display[1] / 2)
                xoffset = 0
                yoffset = 0
                lastX = display[0] / 2
                lastY = display[1] / 2
                initial = False
            else:    
                xoffset = xpos - lastX
                yoffset = ypos - lastY
                lastX = xpos
                lastY = ypos
        #xoffset, yoffset = pygame.mouse.get_rel()
        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= -1
        yoffset *= sensitivity
        yaw += xoffset
        pitch += yoffset
        maxAngle = 89
        if pitch > maxAngle:
            pitch = maxAngle
        if pitch < maxAngle * -1:
            pitch = maxAngle * -1
        frontX = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
        frontY = math.sin(math.radians(pitch))
        frontZ = math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
        cameraFront = Vec3(frontX, frontY, frontZ)
        cameraFront = normalizeVec(cameraFront)
        right = cross(cameraFront, worldUp) #normalized
        view = lookAt(cameraPos, cameraPos + cameraFront, up)
        view = convert(view)
        viewLoc = glGetUniformLocation(shader, "view")
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view)
        projection = Mat4x4()
        fov += fovdelta
        if fov <= 1.0:
            fov = 1.0
            print ("MAX ZOOM REACHED")
        if fov >= 45.0:
            fov = 45.0
        projection = perspective(fov, aspect, 0.1, 100)
        projection = convert(projection)
        projectionLoc = glGetUniformLocation(shader, "projection")
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projection)
        glBindVertexArray(myvao)
        glPointSize(20.0)
        glDrawArrays(GL_POINTS,0,numpos)
        #glDrawElements(GL_POINTS, 3 , GL_UNSIGNED_INT, 0)
        pygame.display.flip()
main()


    

