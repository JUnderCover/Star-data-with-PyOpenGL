from OpenGL.GL import glGetAttribLocation, glUseProgram, \
    glGetUniformLocation, glUniformMatrix4fv, \
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_FALSE

from OpenGL.GL.shaders import compileShader, compileProgram, glDeleteShader

class Shader(object):
    def __init__(self, vertexShaderSource, fragmentShaderSource):
        vertexFile = open(vertexShaderSource, 'r')
        fragmentFile = open(fragmentShaderSource, 'r')
        vertexShader = compileShader(vertexFile.read(), GL_VERTEX_SHADER)
        fragmentShader = compileShader(fragmentFile.read(), GL_FRAGMENT_SHADER)
        self.__shader = compileProgram(vertexShader, fragmentShader)
        vertexFile.close()
        fragmentFile.close()
        glDeleteShader(vertexShader)
        glDeleteShader(fragmentShader)

    def getAttributes(self):
        #the current attributes are position (Vec 3), color (Vec 3) and point size (float)
        #These are inputs to the vertex shader.
        #Each row of data must be structured in this order
        return glGetAttribLocation(self.__shader, "position"),glGetAttribLocation(self.__shader, "vcolor"), glGetAttribLocation(self.__shader, "psize")

    def use(self):
        glUseProgram(self.__shader)

    def setModelMatrix(self, model):
        modelLoc = glGetUniformLocation(self.__shader, "model")
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model)

    def setViewMatrix(self, view):
        viewLoc = glGetUniformLocation(self.__shader, "view")
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view)

    def setProjectionMatrix(self, projection):
        projectionLoc = glGetUniformLocation(self.__shader, "projection")
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projection)

if __name__ == "__main__":
    print("Trying to run Shader class definition?!") 
