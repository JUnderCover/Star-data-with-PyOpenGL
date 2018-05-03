#version 440
in vec3 fcolor;
out vec4 FragColor;
uniform sampler2D tex;
void main()
{
     vec4 texColor = texture(tex, gl_PointCoord);
     FragColor = texColor * vec4(fcolor, 1.0);
}