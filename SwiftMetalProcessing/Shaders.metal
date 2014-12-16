//
//  Shaders.metal
//  SwiftMetalProcessing
//
//  Created by Amund Tveit on 15/12/14.
//  Copyright (c) 2014 Amund Tveit. All rights reserved.

#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    // This calculates sigmoid for _one_ position (=id) in a vector per call on the GPU
    outVector[id] = 1.0 / (1.0 + exp(-inVector[id]));
}
