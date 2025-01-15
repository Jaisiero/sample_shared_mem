#include "daxa/daxa.inl"
#include "shared.inl"
#extension GL_EXT_debug_printf : enable

// Push constant struct
DAXA_DECL_PUSH_CONSTANT(ComputePush, p)

// Shared memory histogram
shared uint[4] histogram;

// Entry point GLSL compute shader
layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint index = gl_GlobalInvocationID.x;

    // Initialize histogram
    histogram[index] = 0;
    barrier();
    
    // Add to histogram
    atomicAdd(histogram[index], 1);
    barrier();

    // Read histogram
    uint count = atomicAdd(histogram[index], 0);
    deref_i(p.global_histograms, index) += count;
    if(count > 0)
        debugPrintfEXT("GLSL: Histogram global bin %u, count %u\n", index,count);
}
