#include "daxa/daxa.inl"
#include "shared.inl"

[[vk::push_constant]] ComputePush p;

// groupshared variables
groupshared Atomic<uint> m[4];

// Entry point with HLSL semantics
[numthreads(4, 1, 1)] void entry_compute_shader(int i : SV_DispatchThreadID)
{
  // global histogram buffer
  Ptr<uint> global_histograms =
      Ptr<uint>(p.global_histograms);

  // Initialize histogram
  m[i] = 0U;
  GroupMemoryBarrierWithGroupSync();

  // Add to histogram
  m[i] += 1U;
  GroupMemoryBarrierWithGroupSync();

  uint count = m[i].load();
  global_histograms[i] += count;
  if(count > 0)
    printf("Histogram global bin %d, count %d\n", i, count);
}