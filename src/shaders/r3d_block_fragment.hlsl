// Port of the CDDA fork's r3d_block.frag, including cutaway rims and alpha testing.
Texture2D<float4> atlas : register(t0, space2);
SamplerState atlas_sampler : register(s0, space2);
struct FragmentInput {
    float2 uv : TEXCOORD0;
    float4 light : TEXCOORD1;
    noperspective float3 ndc : TEXCOORD2;
    float height : TEXCOORD3;
    nointerpolation float4 cut : TEXCOORD4;
    nointerpolation float2 cut2 : TEXCOORD5;
    nointerpolation float occlude : TEXCOORD6;
};
float4 main(FragmentInput input) : SV_Target0 {
    float rim = 1.0;
    if (input.cut.w > 0.0 && input.height > input.cut2.y && input.ndc.z < input.cut.z) {
        const float2 delta = float2(input.ndc.x - input.cut.x, (input.ndc.y - input.cut.y) * input.cut2.x);
        const float distance = length(delta);
        if (distance < input.cut.w) { discard; }
        if (distance < input.cut.w * 1.07) { rim = 0.4; }
    }
    const float4 texel = atlas.Sample(atlas_sampler, input.uv);
    if (texel.a < 0.75) { discard; }
    if (max(input.light.r, max(input.light.g, input.light.b)) < 0.5 / 255.0) {
        if (input.occlude > 0.5) { return float4(0, 0, 0, 1); }
        discard;
    }
    return float4(texel.rgb * input.light.rgb * rim, 1.0);
}
