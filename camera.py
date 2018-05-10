# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2014 mack stone
# Modified by Jai Chauhan 2018



# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import math
class Camera(object):
    def __init__(self, pos=np.array([0.0,0.0,5.0])):
        self.pos = pos
        self.focus = np.array([0.0,0.0,0.0])
        self.camFocus = self.pos - self.focus
        self.normCamFocus = self.camFocus /np.linalg.norm(self.camFocus)
        worldUp = np.array([0.0,1.0,0.0])
        self.right = np.cross(self.normCamFocus, worldUp)
        self.up = np.cross(self.right, self.normCamFocus)
        
    #Matrix methods derived from glm library by Mack Stone
    #See https://github.com/mackst/glm/tree/master/glm
    @staticmethod
    def rotationMatrix(angle, axis):
        rmat = np.zeros((4,4))
        a = math.radians(angle)
        c = math.cos(a)
        s = math.sin(a)
        axis = axis / np.linalg.norm(axis)
        temp = (1. - c) * axis
        rmat[0][0] = c + temp[0] * axis[0]
        rmat[0][1] = 0 + temp[0] * axis[1] + s * axis[2]
        rmat[0][2] = 0 + temp[0] * axis[2] - s * axis[1]

        rmat[1][0] = 0 + temp[1] * axis[0] - s * axis[2]
        rmat[1][1] = c + temp[1] * axis[1]
        rmat[1][2] = 0 + temp[1] * axis[2] + s * axis[0]

        rmat[2][0] = 0 + temp[2] * axis[0] + s * axis[1]
        rmat[2][1] = 0 + temp[2] * axis[1] - s * axis[0]
        rmat[2][2] = c + temp[2] * axis[2]
        return rmat

    @staticmethod
    def lookatMatrix(eye, center, up):
        f = (center - eye) / np.linalg.norm((center - eye))
        cp = np.cross(f, up)
        s = cp / np.linalg.norm(cp)
        u = np.cross(s, f)
        Result = np.diag([1.0,1.0,1.0,1.0])
        Result[0][0] = s[0]
        Result[1][0] = s[1]
        Result[2][0] = s[2]
        Result[0][1] = u[0]
        Result[1][1] = u[1]
        Result[2][1] = u[2]
        Result[0][2] =-f[0]
        Result[1][2] = -f[1]
        Result[2][2] =-f[2]
        Result[3][0] = -1 * np.dot(s, eye)
        Result[3][1] = -1 * np.dot(u, eye)
        Result[3][2] = np.dot(f, eye)
        return Result
