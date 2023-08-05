//
//  Shaders.metal
//  TileMetal01 Shared
//
//  Created by gzonelee on 2023/08/05.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
    float4 shadowPosition;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & u [[ buffer(BufferIndexUniforms) ]],
                               constant ModelUniforms & mu [[ buffer(BufferIndexModelUniforms) ]]
                               )
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = u.projectionMatrix * u.viewMatrix * mu.modelMatrix * position;
    out.texCoord = in.texCoord;
    out.shadowPosition = u.shadowProjectionMatrix * u.shadowViewMatrix * mu.modelMatrix * position;

    return out;
}

vertex ColorInOut vertexShadowShader(Vertex in [[stage_in]],
                               constant Uniforms & u [[ buffer(BufferIndexUniforms) ]],
                               constant ModelUniforms & mu [[ buffer(BufferIndexModelUniforms) ]]
                               )
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = u.shadowProjectionMatrix * u.shadowViewMatrix * mu.modelMatrix * position;
    out.texCoord = in.texCoord;

    return out;
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               constant ModelUniforms & mu [[ buffer(BufferIndexModelUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]],
                               depth2d<float> shadowMap [[ texture(TextureIndexShadow) ]]
                               )
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    constexpr sampler s(
                        coord::normalized, filter::nearest,
                        address::clamp_to_edge,
                        compare_func:: less);
    
    half4 colorSample   = colorMap.sample(colorSampler, in.texCoord.xy);
    
    float3 position
    = in.shadowPosition.xyz / in.shadowPosition.w;
    float2 xy = position.xy;
    xy = xy * 0.5 + 0.5;
    xy.y = 1 - xy.y;
    float shadow_sample = shadowMap.sample(s, xy);
   
    float v = 0;
    if (position.z > shadow_sample + 0.001) {
        v = 0.5;
        return float4(1, 0, 0, 1);
    }
    else {
        v = 1.0;
        return float4(colorSample);
    }
    
//    colorSample *= v;
//    return float4(colorSample.x, colorSample.y, colorSample.z, 1);
}

fragment float4 fragmentFloorShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{

    return float4(0.5, 0.5, 0.7, 1);
}
