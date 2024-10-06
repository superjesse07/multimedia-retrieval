#version 330

uniform mat4 Mvp;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

out vec3 v_vert;
out vec3 v_norm;
flat out vec3 v_flat_norm;
out vec2 v_text;

void main() {
    v_vert = in_position;
    v_norm = in_normal;
    v_flat_norm = in_normal;
    v_text = in_texcoord_0;
    gl_Position = Mvp * vec4(in_position, 1.0);
}