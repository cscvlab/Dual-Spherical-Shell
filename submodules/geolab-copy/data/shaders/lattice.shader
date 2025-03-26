#shader vertex
#version 330 core

uniform vec3 light_pos;
uniform vec3 base_color;
uniform vec3 camera_pos;
uniform float specular;
uniform vec3 ambient_color;
uniform float kd;

out vec3 diffuse_color;
out vec3 specular_color;
out vec2 vertexTexCoord;

uniform mat4 world;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

void main(){
    // vec3 ecPosition = vec3(gl_ModelViewMatrix * gl_Vertex); 
    vec3 tnorm = normalize(normal);
    vec3 lightVec = normalize(light_pos - position);
    vec3 viewVec = normalize(camera_pos - position);
    vec3 Hvec = normalize(viewVec + lightVec);

    float spec = abs(dot(Hvec, tnorm));
    spec = pow(spec, 16.0);

    diffuse_color = base_color * vec3(kd * abs(dot(lightVec, tnorm)));
    diffuse_color = clamp(ambient_color + diffuse_color, 0.0, 1.0);
    specular_color = clamp((base_color * specular * spec), 0.0, 1.0);

    vertexTexCoord[0] = 0;

    // gl_Position = ftransform();

    gl_Position = world * vec4(position.x, position.y, position.z, 1.0f);
}

#shader fragment
#version 330 core

in vec3 diffuse_color;
in vec3 specular_color;
in vec3 vertexTexCoord;

out vec4 color;

uniform vec2 scale;
uniform vec2 threshold;
uniform vec3 surface_color;

void main(){

    // float ss = fract(vertexTexCoord[0].s * scale.s);
    // float tt = fract(vertexTexCoord[0].t * scale.t);

    // if((ss > threshold.s) && (tt > threshold.t))discard;

    vec3 finalColor = surface_color * diffuse_color + specular_color;

    color = vec4(finalColor, 1.0);

}