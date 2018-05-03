#version 440
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