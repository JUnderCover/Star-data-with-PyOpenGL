import pygame
import random
import numpy
import numpy as np
from numpy.linalg import inv
import math
import sys
import matplotlib.cm as cm
import matplotlib as mpl
from ctypes import *
from PIL import Image

from glm.gtc.matrix_transform import *
from glm.detail.type_vec4 import *
from glm.detail.type_mat4x4 import *

import OpenGL.GL.shaders
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# import yt
# yt.enable_plugins()

# ds = yt.load('DD0220/output_0220')
# filters = ['p3_sn', 'p3_bh', 'p3_living', 'p2', 'stars', 'dm']
# for f in filters:
#     ds.add_particle_filter(f)

# star_type = 'p2'
# ad = ds.all_data()

# position = ad[star_type, 'particle_position']
# position = (np.array(position) - 0.5) * 2
# age = ad[star_type, 'age']
# age = np.array(age.in_units('Myr'))
# mass = ad[star_type, 'particle_mass']
# mass = np.array(mass.in_units('Msun'))


def getPoint(near, far, point):
    point = np.array(point)
    ray = far - near
    ray_dot = np.dot(ray, ray)
    p = point - near
    p_ray_dot = np.dot(p, ray)
    t = p_ray_dot / ray_dot
    q = near + ray * t
    return q

def convertnp(matrix):
    vecList = matrix._Mat4x4__value
    m = np.zeros((4,4))
    trans = []
    for i, vec in enumerate(vecList):
        m[i,0] = vec.x
        m[i,1] = vec.y
        m[i,2] = vec.z
        m[i,3] = vec.w
        #trans.append(vec.x)
        #trans.append(vec.y)
        #trans.append(vec.z)
        #trans.append(vec.w)
    return m




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
    # positions = []
    # myfile = open("p2data.txt", "r")
    # for line in myfile:
    #     positions.append((float(line[2:12]) - 0.5) * 2)
    #     positions.append((float(line[14:24]) - 0.5) * 2)
    #     positions.append((float(line[26:36]) - 0.5) * 2)
    # positions = np.array(positions, dtype=np.float32)
    
    data = np.loadtxt('p2data_v2.txt', dtype=np.float32)
    data[:, 0:3] = (data[:, 0:3] - 0.5) * 2
    #Fake Point
    data[0,0] = -0.2
    data[0,1] = 0.1
    data[0,2] = -1.0
    myp = np.array([data[0,0], data[0,1], data[0,2]])
    data[0,4] = 900
    #End Fake Point
    pos = data[:, 0:3]
    norm = mpl.colors.Normalize(vmin=1, vmax=1000)
    data = np.append(data, cm.bwr(norm(data[:,4])), axis=1)
    size0 = np.min(data[:, 3])
    mass = size0 * np.sqrt(data[:, 3] / 30000)
    mass = mass.reshape(mass.shape[0], 1)
    mass = np.maximum(mass, 1)
    mass[0] = 20
    data = np.delete(data, 3, 1)  # drop mass
    data = np.delete(data, 3, 1)  # drop age
    data = np.delete(data, 6, 1)  # drop Alpha
    #m = np.ones((data.shape[0], 1))
    #m = m * 80.0
    #m[0::2] = 20.0
    data = numpy.hstack((data, mass))
    data = data.astype(numpy.float32)

    #yt region
    # norm = mpl.colors.Normalize(vmin=1, vmax=1000)
    # colors = cm.bwr(norm(age))
    # colors = colors[:, 0:-1]
    # size0 = np.min(mass)
    # m = np.array(mass) / 30000
    # m = size0 * np.sqrt(m)
    # m = np.maximum(m, 1)
    # m = m.reshape(m.shape[0], 1)
    # data = np.hstack((position, colors, m))
    # data = data.astype(np.float32)

    



    
    pygame.init()
    #pygame.mouse.set_cursor(*pygame.cursors.diamond)
    display = (1000, 800)
    #display = (800, 600)
    aspect = display[0] / display [1]
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    myvao = glGenVertexArrays(1)
    glBindVertexArray(myvao)
    myvbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, myvbo)
    floatsize = 4  # 4 bytes, 32 bits
    size = floatsize * data.size
    glBufferData(GL_ARRAY_BUFFER, size, numpy.ravel(data), GL_STATIC_DRAW)

    glEnable(GL_PROGRAM_POINT_SIZE)
    vertexShaderSource = """
    #version 330
    in vec3 position;
    in vec3 vcolor;
    in float psize;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    uniform vec2 screenSize;
    uniform float spriteSize;

    out vec3 fcolor;
    


    void main()
    {
        gl_Position = projection * view * model *  vec4(position, 1.0);
        gl_PointSize = psize;
        fcolor = vcolor;
    }
    """
    fragmentShaderSource = """
    #version 330
    in vec3 fcolor;
    out vec4 FragColor;
    uniform sampler2D tex;
    void main()
    {
        vec4 texColor = texture(tex, gl_PointCoord);
        //texColor.a = 1.0 - texColor.r;
        FragColor = texColor * vec4(fcolor, 1.0);
    }
    """
    shader = shaders.compileProgram(shaders.compileShader(vertexShaderSource, GL_VERTEX_SHADER), shaders.compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER))
    positionAttrib = glGetAttribLocation(shader, "position")
    colorAttrib = glGetAttribLocation(shader, "vcolor")
    sizeAttrib = glGetAttribLocation(shader, "psize")
    #ctypes.c_void_p(offset)
    stride = 7 * floatsize
    glVertexAttribPointer(positionAttrib, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glVertexAttribPointer(colorAttrib, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * floatsize))
    glVertexAttribPointer(sizeAttrib, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * floatsize))
    glEnableVertexAttribArray(positionAttrib)
    glEnableVertexAttribArray(colorAttrib)
    glEnableVertexAttribArray(sizeAttrib)
    numpos = int(data.shape[0])



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
    
    #yaw = -90.0
    yaw = 0.0
    pitch = 0.0
    xoffset = 0.0
    yoffset = 0.0
    lastX = display[0] / 2
    lastY = display[1] / 2
    fov = 45.0

    #Initialize matrix:
    projection = perspective(fov, aspect, 0.1, 100)
    view = Mat4x4()
    model = Mat4x4()
    
    #Look at Vectors
    worldUp = Vec3(0.0, 1.0, 0.0)
    origin = Vec3(0.0, 0.0, 0.0)
    cameraPos = Vec3(0.0, 0.0, 5.0)
    focus = Vec3(0.0, 0.0, 0.0)
    frontX = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
    frontY = math.sin(math.radians(pitch))
    frontZ = math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
    cameraFront = Vec3(frontX, frontY, frontZ)
    cameraFront = normalizeVec(cameraFront)
    right = cross(cameraFront, worldUp)
    up = cross(right, cameraFront) #normalized internally
    delta = 0.0
    delta2 = 0.0
    d = 0.0
    fovdelta = 0.0
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(True)
    middle = False
    select = False

    #glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_COLOR)
    #glBlendFunc(GL_SRC_ALPHA, GL_ZERO)

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
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and select:
                    print("Point Selected !?")
                if event.button == 4:
                    # ZOOM IN
                    if middle:
                        print("ZOOM IN")
                        fovdelta = -.2
                if event.button == 5:
                    # ZOOM OUT
                    if middle:
                        print("ZOOM OUT")
                        fovdelta = 0.2
                if event.button == 2:
                    middle = True
                if event.button == 3:
                    #Reset initial conditions
                    deltaTime = 0.0
                    lastFrame = 0.0

                    initial = True
                    #yaw = -90.0
                    yaw = 0.0
                    pitch = 0.0
                    xoffset = 0.0
                    yoffset = 0.0
                    lastX = display[0] /  2
                    lastY = display[1] / 2
                    fov = 45.0

                    projection = perspective(fov, aspect, 0.1, 100)
                    view = Mat4x4()
                    model = Mat4x4()


                    #Look at Vectors
                    worldUp = Vec3(0.0, 1.0, 0.0)
                    origin = Vec3(0.0, 0.0, 0.0)
                    cameraPos = Vec3(0.0, 0.0, 5.0)
                    focus = Vec3(0.0, 0.0, 0.0)
                    frontX = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
                    frontY = math.sin(math.radians(pitch))
                    frontZ = math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
                    cameraFront = Vec3(frontX, frontY, frontZ)
                    cameraFront = normalizeVec(cameraFront)
                    right = cross(cameraFront, worldUp)
                    up = cross(right, cameraFront) #normalized internally    
                    delta = 0.0
                    delta2 = 0.0
                    d = 0.0
                    fovdelta = 0.0
                    middle = False
                    select = False
                    initial = True
                    model = Mat4x4()
                    newmodel = Mat4x4()
                    oldmodel = Mat4x4()

                    xoffset = 0.0
                    yoffset = 0.0
                    
                    #Right Click to exit
                    #pygame.quit()
                    #quit()
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    middle = False
                    print("DONE")
                    fovdelta = 0
            if event.type == pygame.KEYDOWN:
                if event.key == K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_LEFT:
                    delta = right * 0.01
                    #delta = cross(cameraFront, up) * cameraSpeed * -1
                    #cameraPos -= cross(cameraFront, up) * cameraSpeed
                if event.key == pygame.K_RIGHT:
                    delta = right * 0.01 * -1
                    #delta = cross(cameraFront, up) * cameraSpeed
                    #cameraPos += cross(cameraFront, up) * cameraSpeed
                if event.key == pygame.K_UP:
                    d = 0.01
                    delta2= 0.01 * -1 * camFocus 
                    #delta = cameraSpeed * cameraFront
                    #cameraPos += cameraSpeed * cameraFront
                if event.key == pygame.K_DOWN:
                    d = 0.01 * -1
                    delta2= 0.01 * camFocus
                    #delta = cameraSpeed * cameraFront * -1
                    #cameraPos -= cameraSpeed * cameraFront
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    #print(delta)
                    delta = Vec3(0.0, 0.0, 0.0)
                if event.key == pygame.K_RIGHT:
                    #print(delta)
                    delta = Vec3(0.0, 0.0, 0.0)
                if event.key == pygame.K_UP:
                    d = 0.0
                    delta2 = Vec3(0.0, 0.0, 0.0)
                if event.key == pygame.K_DOWN:
                    d = 0.0
                    delta2 = Vec3(0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        #glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader)
        glBindTexture(GL_TEXTURE_2D, textureobject)


        #Ray Tracing
        xpos, ypos = pygame.mouse.get_pos()
        #OpenGL Viewport, lower left is (0,0)
        #In Window Coordinates top left is (0,0)
        ypos = display[1] - ypos
        #xpos = (xpos - (display[0] / 2))
        #ypos = ((display[1] / 2) - ypos)
        #viewport = np.zeros((4, 1))
        viewport = np.array(glGetIntegerv( GL_VIEWPORT))
        m = convertnp(model)
        v = convertnp(view)
        mv = np.matmul(v, m)
        p = convertnp(projection)
        xyz_n = np.array(gluUnProject(xpos, ypos, 0.0, mv, p, viewport))
        xyz_f = np.array(gluUnProject(xpos, ypos, 1.0, mv, p, viewport))

        #Line Segment xyz_n and xyz_f
        #Pos: all points
        #Q: Points on line segment closest all points in Pos
        ray = xyz_f - xyz_n
        ray_dot = np.dot(ray, ray)
        p = pos - xyz_n
        p_ray_dot = np.dot(p, ray)
        t = p_ray_dot / ray_dot
        z = np.matmul(np.reshape(t, (t.shape[0], 1)), np.reshape(ray, (3,1)).T)
        q = xyz_n + z
        dist = np.linalg.norm(q - pos, axis = 1)
        mindist = np.min(dist)
        #print(mindist)
        if(mindist < 0.02):
            pygame.mouse.set_cursor(*pygame.cursors.diamond)
            select = True
        else:
            pygame.mouse.set_cursor(*pygame.cursors.arrow)
            select = False
        #pLine = getPoint(xyz_n, xyz_f, myp)
        #dist = pLine - myp
        #x = np.linalg.norm(dist)
        # print(x)
        # if(x < 0.02):
        #     z = 5
        #     pygame.mouse.set_cursor(*pygame.cursors.diamond)
        # else:
        #     pygame.mouse.set_cursor(*pygame.cursors.arrow)

        model = Mat4x4()
        modelList = convert(model)
        modelLoc = glGetUniformLocation(shader, "model")
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, modelList)
        view = Mat4x4()
        #view = translate(view, Vec4(0.0, 0.0, -3.0, 0.0))
        #view = translate(view, Vec4(0.739, 0.637, -0.7, 0.0))
        
        #Rotation Around Cluster
        #radius = 0.2
        #camX = -0.739 + math.sin(timeValue) * radius
        #camZ = 0.638  + math.cos(timeValue) * radius
        #cameraPos = Vec3(camX, -0.637, camZ)
        #view = lookAt(cameraPos, cameraTarget, up)

        #Calculate Mouse Offsets
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
            oldV = Mat4x4()
        else:  
            xoffset = xpos - lastX
            yoffset = ypos - lastY
            lastX = xpos
            lastY = ypos 
        #xoffset, yoffset = pygame.mouse.get_rel()
        sensitivity = 0.3
        xoffset *= sensitivity
        yoffset *= -1
        yoffset *= sensitivity
        yaw = xoffset
        pitch = yoffset
        maxAngle = 89
        if pitch > maxAngle:
            pitch = maxAngle
        if pitch < maxAngle * -1:
            pitch = maxAngle * -1
        #cameraPos += delta
        #cameraPos = cameraPos + (d * cameraFront)
        frontX = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
        frontY = math.sin(math.radians(pitch))
        frontZ = math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
        #cameraFront = Vec3(frontX, frontY, frontZ)
        
        #cameraFront = near - cameraPos

        
        # print(near)
        # print(loc)
        #cameraFront = normalizeVec(cameraFront)
        #right = cross(cameraFront, worldUp) #normalized
        #up = cross(right, cameraFront)
        # # view = lookAt(cameraPos, normalizeVec(near), up)
        #view = lookAt(cameraPos, cameraPos + cameraFront, up)
        #viewList = convert(view)
        #viewLoc = glGetUniformLocation(shader, "view")
        #glUniformMatrix4fv(viewLoc, 1, GL_FALSE, viewList)
        #print(cameraSpeed)
        focus += delta
        cameraPos += delta + delta2
        #print(cameraPos)
        camFocus = cameraPos - focus
        right = cross(normalizeVec(camFocus), worldUp)
        #print(focus)
        up = cross(right, normalizeVec(camFocus))
        r = Mat4x4()
        r = rotate(r, yaw, up)
        r = rotate(r, pitch, right)
        r = convertnp(r)

        cf = np.array([camFocus.x, camFocus.y, camFocus.z, 1])
        cf = np.matmul(r, cf.T)
        cf = Vec3(cf[0], cf[1], cf[2])
        new_cameraPos = cf + focus
        view = lookAt(new_cameraPos, focus, up)
        cameraPos = new_cameraPos
        viewList = convert(view)
        viewLoc = glGetUniformLocation(shader, "view")
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, viewList)

        #Projection
        projection = Mat4x4()
        fov += fovdelta
        if fov <= 1.0:
            fov = 1.0
            print ("MAX ZOOM REACHED")
        if fov >= 45.0:
            fov = 45.0
        projection = perspective(fov, aspect, 0.1, 100)
        projectionList = convert(projection)
        projectionLoc = glGetUniformLocation(shader, "projection")
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projectionList)
        glBindVertexArray(myvao)
        #glPointSize(20.0)
        glDrawArrays(GL_POINTS,0,numpos)
        #glDrawElements(GL_POINTS, 3 , GL_UNSIGNED_INT, 0)
        pygame.display.flip()
main()
