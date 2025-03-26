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
    vertexNormal = normal;
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
uniform float intensity;

void main(){
    vec3 e = normalize(vertexPosition - camera_pos);
    vec3 n = normalize(vertexNormal);

    vec3 r = vertexPosition - light_pos;
    float d2 = r.x*r.x + r.y*r.y + r.z*r.z;
    vec3 l = normalize(r);
    vec3 h = normalize(e + r);

    // ambient
    vec3 amb = ambient_color;
    // diffuse
    vec3 dif = base_color / d2 * intensity * clamp(dot(n, l), 0.0, 1.0);
    // specular
    vec3 spe = specular * base_color / d2 * clamp(dot(n, h), 0.0, 1.0);

    vec3 c = clamp(amb + dif + spe, 0.0f, 1.0f);
    // vec3 c = clamp(spe, 0.0f, 1.0f);
    color = vec4(c, 1.0f);

}