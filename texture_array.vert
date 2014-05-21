#version 330 
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 texcoord;

// uniform varaibles
uniform mat4 mv_mat;
uniform mat4 mp_mat;
uniform float weight;

// to fragment shader
out vec3 frag_tex;
out float frag_weight;

void main()
{
    gl_Position = mp_mat * mv_mat * vec4(position,1.0);

	frag_tex = texcoord;
	frag_weight = weight;
}