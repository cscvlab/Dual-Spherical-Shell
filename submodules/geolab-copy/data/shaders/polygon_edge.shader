#shader vertex
#version 330 core

uniform mat4 world;
layout(location = 0) in vec3 position;
out vec3 vertexNormal;

void main(){
    gl_Position = world * vec4(position.x, position.y, position.z, 1.0f); 
}

#shader fragment
#version 330 core

out vec4 color;
uniform vec3 fcolor;

void main(){
    color = vec4(fcolor, 1.0f);
}