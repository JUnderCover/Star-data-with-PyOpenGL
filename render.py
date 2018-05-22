import pygame
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from ctypes import *
from PIL import Image


from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from shader import Shader
from camera import Camera

#Uncomment if using yt (1/2)
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


def main():
    data = np.loadtxt('p2data_v2.txt', dtype=np.float32)
    data[:, 0:3] = (data[:, 0:3] - 0.5) * 2
    pos = data[:, 0:3]
    norm = mpl.colors.Normalize(vmin=1, vmax=1000)
    data = np.append(data, cm.bwr(norm(data[:,4])), axis=1)
    size0 = np.min(data[:, 3])
    mass = size0 * np.sqrt(data[:, 3] / 30000)
    mass = mass.reshape(mass.shape[0], 1)
    mass = np.maximum(mass, 1)
    data = np.delete(data, 3, 1)  # drop mass
    data = np.delete(data, 3, 1)  # drop age
    data = np.delete(data, 6, 1)  # drop Alpha
    data = np.hstack((data, mass))
    data = data.astype(np.float32)

    # #Uncomment if Using yt (2/2)
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
    # pos = data[:, 0:3]
    
    pygame.init()
    display = (1000, 800)
    #display = (800, 600)
    aspect = display[0] / display [1]
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    #Initialize VAO an VBO
    myvao = glGenVertexArrays(1)
    glBindVertexArray(myvao)
    myvbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, myvbo)
    floatsize = 4  # 4 bytes, 32 bits
    size = floatsize * data.size
    glBufferData(GL_ARRAY_BUFFER, size, np.ravel(data), GL_STATIC_DRAW)
    glEnable(GL_PROGRAM_POINT_SIZE)

    #Shader Compilation, Attributes and set stride
    myshader = Shader("vertexShader.glsl", "fragmentShader.glsl")
    positionAttrib, colorAttrib, sizeAttrib = myshader.getAttributes()
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
    #texture
    textureobject = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureobject)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    image = Image.open("halo.jpg")
    width, height = image.size
    print (width, height)
    img_data = np.array(list(image.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    
    #Camera
    mycamera = Camera()
    
    #Look at Variables
    yaw = 0.0
    pitch = 0.0
    xoffset = 0.0
    yoffset = 0.0
    lastX = display[0] / 2
    lastY = display[1] / 2

    #Initialize matrix:
    zNear = 0.1
    zFar = 100
    projection = Camera.perspectiveMatrix(mycamera.fov, aspect, zNear, zFar)
    view = np.diag([1.0,1.0,1.0,1.0])
    model = np.diag([1.0,1.0,1.0,1.0])
    
    #Deltas
    delta = 0.0
    delta2 = 0.0
    fovdelta = 0.0
    #Booleans
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(True)
    middle = False
    select = False
    center = False
    rotate_mode = True
    setOrigin = False
    newFocus = False
    initial = True

    #glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_COLOR)
    #glBlendFunc(GL_SRC_ALPHA, GL_ZERO)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == K_r:
                    if(not rotate_mode):
                        initial = True
                    rotate_mode = not rotate_mode
                if event.key == K_o and newFocus:
                    setOrigin = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and select:
                    center = True
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
                if event.button == 3: #Right Click to Reset
                    #Camera
                    mycamera = Camera()
                    
                    #Look at Variables
                    yaw = 0.0
                    pitch = 0.0
                    xoffset = 0.0
                    yoffset = 0.0
                    lastX = display[0] / 2
                    lastY = display[1] / 2

                    #Initialize matrix:
                    projection = Camera.perspectiveMatrix(mycamera.fov, aspect, zNear, zFar)
                    view = np.diag([1.0,1.0,1.0,1.0])
                    model = np.diag([1.0,1.0,1.0,1.0])

                    #Deltas
                    delta = 0.0
                    delta2 = 0.0
                    fovdelta = 0.0
                    #Booleans
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(True)
                    middle = False
                    select = False
                    center = False
                    rotate_mode = True
                    setOrigin = False
                    newFocus = False
                    initial = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    middle = False
                    print("DONE")
                    fovdelta = 0
            if event.type == pygame.KEYDOWN:
                if event.key == K_q: #Q to Quit
                    pygame.quit()
                    quit()
                if event.key == pygame.K_LEFT:
                    delta = mycamera.right * 0.01
                if event.key == pygame.K_RIGHT:
                    delta = mycamera.right * 0.01 * -1
                if event.key == pygame.K_UP:
                    delta2 = 0.005 * -1 * mycamera.camFocus
                if event.key == pygame.K_DOWN:
                    delta2 = 0.005 * mycamera.camFocus
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    delta = np.array([0.0, 0.0, 0.0])
                if event.key == pygame.K_RIGHT:
                    delta = np.array([0.0, 0.0, 0.0])
                if event.key == pygame.K_UP:
                    delta2 = np.array([0.0, 0.0, 0.0])
                if event.key == pygame.K_DOWN:
                    delta2 = np.array([0.0, 0.0, 0.0])
        glClear(GL_COLOR_BUFFER_BIT)
        #glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        myshader.use()
        glBindTexture(GL_TEXTURE_2D, textureobject)
        
        #Ray Tracing
        xpos, ypos = pygame.mouse.get_pos()
        #OpenGL Viewport, lower left is (0,0)
        #In Window Coordinates top left is (0,0)
        ypos = display[1] - ypos
        viewport = np.array(glGetIntegerv( GL_VIEWPORT))
        mv = np.matmul(view, model)
        xyz_n = np.array(gluUnProject(xpos, ypos, 0.0, mv, projection, viewport))
        xyz_f = np.array(gluUnProject(xpos, ypos, 1.0, mv, projection, viewport))

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
        closest_point = pos[np.argmin(dist), :]
        diff = mycamera.pos - mycamera.focus
        cameraToFocus = np.linalg.norm(np.array([diff[0], diff[1], diff[2]]))
        thresh = 0.004 * cameraToFocus
        if(mindist < thresh):
            pygame.mouse.set_cursor(*pygame.cursors.diamond)
            select = True
        else:
            pygame.mouse.set_cursor(*pygame.cursors.arrow)
            select = False

        #Set Model Matrix
        model = np.diag([1.0,1.0,1.0,1.0])
        myshader.setModelMatrix(model)

        #View Matrix Calculation
        #Calculate Mouse Offsets
        if(rotate_mode):
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
        else:
            xoffset = 0.0
            yoffset = 0.0
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
            
        mycamera.focus += delta
        mycamera.pos += delta + delta2
        if setOrigin:
            mycamera.focus = np.array([0.0, 0.0, 0.0])
            pygame.mouse.set_pos(display[0] / 2, display[1] / 2)
            lastX = display[0] / 2
            lastY = display[1] / 2
            newFocus = False
            setOrigin = False
        if center:
            pygame.mouse.set_pos(display[0] / 2, display[1] / 2)
            mycamera.focus = closest_point
            lastX = display[0] / 2
            lastY = display[1] / 2
            center = False
            newFocus = True
        mycamera.translate(delta, delta2)
        r1 = Camera.rotationMatrix(yaw, mycamera.up)
        r2 = Camera.rotationMatrix(pitch, mycamera.right)
        rot = np.matmul(r2, r1)
        cf = np.append(mycamera.camFocus , 1)
        cf = np.matmul(rot, cf.T)
        cf = cf[0:3]  # drop appended 4th. element
        mycamera.pos = cf + mycamera.focus
        view = Camera.lookatMatrix(mycamera.pos, mycamera.focus, mycamera.up)
        myshader.setViewMatrix(view)
        
        #Projection Matrix
        mycamera.fov += fovdelta
        if mycamera.fov <= 1.0:
            mycamera.fov = 1.0
            print ("MAX ZOOM REACHED")
        if mycamera.fov >= 45.0:
            mycamera.fov = 45.0
        projection = Camera.perspectiveMatrix(mycamera.fov, aspect, 0.1, 100)
        myshader.setProjectionMatrix(projection)
        #Rendering
        glBindVertexArray(myvao)
        glDrawArrays(GL_POINTS,0,numpos)
        pygame.display.flip()


if __name__ == "__main__":
    main()
