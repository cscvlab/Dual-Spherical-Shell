#shader vertex
#version 330 core

uniform mat4 world;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

void main(){
    gl_Position = world * vec4(position.x, position.y, position.z, 1.0f); 
}

#shader fragment
#version 330 core

uniform vec4 in_color;
out vec4 color;

void main(){
    color = in_color;
}