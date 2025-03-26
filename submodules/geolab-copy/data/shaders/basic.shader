#shader vertex
#version 330 core

uniform mat4 world;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;
out vec3 vertexNormal;
out vec2 vertexTexCoord;
out vec3 vertexPosition;

void main(){
    gl_Position = world * vec4(position.x, position.y, position.z, 1.0f); 
    vertexNormal = normalize(normal);
    vertexTexCoord = tex_coord;
    vertexPosition = position;
}

#shader fragment
#version 330 core

in vec3 vertexNormal;
in vec2 vertexTexCoord;
in vec3 vertexPosition;
out vec4 color;

uniform vec3 light_pos;
uniform vec3 ambient_color;
uniform vec3 base_color;

uniform vec3 camera_pos;
uniform int p;
uniform float specular;

void main(){
    vec3 c = 0.5 * vertexNormal + 0.5;
    color = vec4(c, 1.0f);
}