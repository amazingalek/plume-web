// Link with --pre-js for plume's WebGPU backend. Requesting a WebGPU device is asynchronous, so this does it before
// main() runs and hands the device over as Module.preinitializedWebGPUDevice, which emscripten_webgpu_get_device()
// returns. Optional features plume can use are asked for when the adapter has them. Validation errors and a lost
// device are logged to the console.
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

    const optionalFeatures = ['float32-filterable', 'depth-clip-control', 'dual-source-blending', 'texture-compression-bc', 'timestamp-query'];
    const requiredFeatures = optionalFeatures.filter((feature) => adapter.features.has(feature));
    const device = await adapter.requestDevice({ requiredFeatures });
    device.addEventListener('uncapturederror', (event) => console.error('WebGPU error: ' + event.error.message));
    device.lost.then((info) => console.error('WebGPU device lost: ' + info.message));
    Module['preinitializedWebGPUDevice'] = device;
    removeRunDependency('plume-webgpu-device');
  })().catch((error) => {
    console.error(error);
    abort(error);
  });
});
