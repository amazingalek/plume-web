// Link with --pre-js for plume's WebGPU backend. Requesting a WebGPU device is asynchronous, so this does it before
// main() runs and hands the device over as Module.preinitializedWebGPUDevice, which emscripten_webgpu_get_device()
// returns. Optional features plume can use are asked for when the adapter has them, and the adapter's own maximum for
// the limits a renderer with big descriptor sets and buffers runs into first (WebGPU's defaults are far below what
// desktop GPUs offer). Validation errors and a lost device are logged to the console. The adapter's info (vendor,
// architecture) is kept as Module.plumeWebGPUAdapterInfo. At the end, setBindGroup is made to work in Firefox with a
// heap past 2 GB.
Module['preRun'] = [].concat(Module['preRun'] || []);
Module['preRun'].push(() => {
  addRunDependency('plume-webgpu-device');
  (async () => {
    if (!navigator.gpu) {
      throw new Error('This browser has no WebGPU (navigator.gpu is missing; the page must be a secure context).');
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('No WebGPU adapter.');
    }

    const optionalFeatures = ['float32-filterable', 'float32-blendable', 'depth-clip-control', 'dual-source-blending', 'texture-compression-bc', 'timestamp-query'];
    const requiredFeatures = optionalFeatures.filter((feature) => adapter.features.has(feature));
    const raisedLimits = ['maxSampledTexturesPerShaderStage', 'maxSamplersPerShaderStage', 'maxStorageBuffersPerShaderStage',
      'maxStorageTexturesPerShaderStage', 'maxBindingsPerBindGroup', 'maxStorageBufferBindingSize', 'maxBufferSize'];
    const requiredLimits = {};
    for (const limit of raisedLimits) {
      if (adapter.limits[limit] !== undefined) {
        requiredLimits[limit] = adapter.limits[limit];
      }
    }

    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    device.addEventListener('uncapturederror', (event) => console.error('WebGPU error: ' + event.error.message));
    device.lost.then((info) => console.error('WebGPU device lost: ' + info.message));
    Module['preinitializedWebGPUDevice'] = device;
    Module['plumeWebGPUAdapterInfo'] = adapter.info;
    removeRunDependency('plume-webgpu-device');
  })().catch((error) => {
    console.error(error);
    abort(error);
  });
});

// emdawnwebgpu hands setBindGroup its dynamic offsets as the whole wasm heap (HEAPU32) with a start and a length.
// Firefox refuses any ArrayBuffer or view larger than 2 GB as an argument, so on a page whose heap has grown past that
// every bind group with dynamic offsets threw and nothing was drawn. A view that big is replaced by a copy of only the
// offsets; Chrome, and any heap under 2 GB, take the call as before. (writeBuffer and writeTexture already pass a
// subarray.)
for (const name of ['GPUComputePassEncoder', 'GPURenderPassEncoder', 'GPURenderBundleEncoder']) {
  const encoder = globalThis[name];
  if (!encoder) {
    continue;
  }

  const setBindGroup = encoder.prototype.setBindGroup;
  encoder.prototype.setBindGroup = function (index, bindGroup, offsets, start, length) {
    if (offsets && offsets.byteLength > 0x7fffffff) {
      return setBindGroup.call(this, index, bindGroup, offsets.slice(start, start + length));
    }

    return setBindGroup.apply(this, arguments);
  };
}
