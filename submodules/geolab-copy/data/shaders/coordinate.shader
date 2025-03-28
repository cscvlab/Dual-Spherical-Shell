#shader vertex
#version 330 core

uniform mat4 world;
layout(location = 0) in vec4 position;
layout(location = 1) in vec3 color;
out vec3 vertexColor;

void main(){
    gl_Position = world * vec4(position.x, position.y, position.z, 1.0f); 
    vertexColor = color;
}

#shader fragment
#version 330 core

in vec3 vertexColor;
out vec4 color;

void main(){
    color = vec4(vertexColor, 1.0f);
}