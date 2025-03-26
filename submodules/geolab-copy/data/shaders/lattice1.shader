#shader vertex
#version 330 core

uniform mat4 world;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

out vec3 vertexPosition;
out vec3 vertexNormal;

void main(){
    gl_Position = world * vec4(position.x, position.y, position.z, 1.0f);
    vertexPosition = position;
    vertexNormal = normal;
}

#shader fragment
#version 330 core

uniform vec3 light_pos;
uniform vec3 base_color;
uniform vec3 camera_pos;
uniform float specular;
uniform vec3 ambient_color;
uniform float kd;

in vec3 vertexPosition;
in vec3 vertexNormal;

out vec4 color;

uniform vec2 scale;
uniform vec2 threshold;
uniform vec3 surface_color;

void main(){

    vec3 tnorm = normalize(vertexNormal);
    vec3 lightVec = normalize(light_pos - vertexPosition);
    vec3 viewVec = normalize(camera_pos - vertexPosition);
    vec3 Hvec = normalize(viewVec + lightVec);

    float spec = abs(dot(Hvec, tnorm));
    spec = pow(spec, 16.0);

    vec3 diffuse_color = base_color * vec3(kd * abs(dot(lightVec, tnorm)));
    diffuse_color = clamp(ambient_color + diffuse_color, 0.0, 1.0);
    vec3 specular_color = clamp((base_color * specular * spec), 0.0, 1.0);

    vec3 finalColor = surface_color * diffuse_color + specular_color;

    color = vec4(finalColor, 1.0);

}