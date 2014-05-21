#version 330
#extension GL_EXT_texture_array: enable

uniform sampler2DArray tex_array;
uniform int TexNum;

in vec3 frag_tex;
in float frag_weight;

out vec4 frag_color;
void main()
{
    float actual_layer = max(0, min(TexNum-1, floor(frag_tex.z+0.5)));
    vec3 texCoord = vec3(frag_tex.xy,actual_layer);
	frag_color = frag_weight * texture2DArray(tex_array,texCoord.xyz);
}