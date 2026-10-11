//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

// WebGPU backend, for the browser through Emscripten and Emdawnwebgpu (--use-port=emdawnwebgpu).
//
// What core WebGPU lacks, and what this backend does instead:
// - Push constants: a shader has one push constant block, as in Vulkan, with each range at its offset in it. Here
//   the block is a uniform buffer at @group(3) @binding(999) (PushConstantsGroup, PushConstantsBinding: the last of
//   the 4 bind groups WebGPU guarantees), written into a per command list ring and bound with a dynamic offset. A
//   layout with 4 descriptor sets shares group 3 between its 4th set and the block, so the set can't use binding
//   999; with fewer, groups between the layout's sets and the block are bound empty. WGSL made from SPIR-V by naga
//   declares the block as var<immediate> (var<push_constant> before naga 30); PlumeWGSL.cmake rewrites that line.
// - Bind group layouts need each texture's sample type and each sampler's filtering, which plume's descriptor
//   ranges don't carry. So a descriptor set derives its layout from what is bound in it, and a pipeline is created
//   at draw time for the layouts of the sets bound then, and cached per pipeline layout.
// - Buffer mapping is asynchronous. An UPLOAD buffer keeps a CPU copy: map() returns it and unmap() writes it to the
//   GPU, and a buffer left mapped is written in full on every executeCommandLists(), since a persistently mapped
//   D3D12-style upload buffer never says what changed. A READBACK buffer that a submitted command list copied into
//   is mapped asynchronously right after the submission, and the callback copies it into the CPU copy, which map()
//   returns, and isMapReady() is false until it has run. Behind plume_device_thread.h (a build with threads), the
//   calling thread polls isMapReady() while the device thread is back in its event loop, where the callback runs.
//   Without threads, map() waits for it itself when the build can (ASYNCIFY or JSPI: emscripten_sleep() hands the
//   thread back to its event loop). Either way execute, wait, map reads what the GPU wrote, as on the other backends.
//   A build that can't wait gets whatever arrived before the call, possibly a submission old.
// - Formatted buffer views (texel buffers) don't exist. A formatted view binds its buffer as a read-only storage
//   buffer of raw bytes, the binding padded to a multiple of 4, so the shader reads array<u32> and unpacks R8 or R16
//   elements itself: element i of an n-byte format is at byte i * n, little-endian. A writable formatted view of
//   elements under 4 bytes shares each u32 between neighbours, so a shader writing it needs atomicOr/atomicAnd.
// - A command list is recorded straight into a WGPUCommandEncoder. Render passes open on the first draw or clear
//   after setFramebuffer() and close on anything that can't happen inside one (copies, dispatches, barriers). A clear
//   of a whole target is the load op of the next pass, and a pass with only clears in it is opened when it closes. A
//   clear of rectangles draws a triangle over the target clipped to each rectangle, with a pipeline of the backend's
//   own: the color is the blend constant, the depth both ends of the viewport, the stencil value the reference.
//   Integer color targets can't blend, nor 32-bit float ones without float32-blendable, so rectangle clears of
//   those are ignored.
// - Resolving part of a texture, MSAA sample locations, timestamp queries, geometry shaders, binding arrays,
//   triangle fans, logic ops and dynamic depth bias are not supported, and the capabilities say so where plume has
//   one (resolveRegion, sampleLocations, queryPools, triangleFan, dynamicDepthBias).
// - Fences and semaphores are no-ops: WebGPU orders submissions and writeBuffer() calls on its one queue, and a
//   READBACK buffer waits for its own data in map().
// - The browser shows the canvas once the thread that drew it returns to its event loop. When the build can wait,
//   present() does that until the next animation frame, which paces presents to the display, keeps at most one frame
//   in flight, and times the display for getRefreshRate(). Otherwise present() returns at once and the caller has to
//   give the thread back to the browser itself (emscripten_set_main_loop or similar), and the rate reads 60. Behind
//   plume_device_thread.h the device thread does that between any two of its tasks, and presents aren't paced.
// - Every call must come from the thread that owns the device's JavaScript object: WebGPU objects can't be shared
//   between workers.
// - Shaders are WGSL text (RenderShaderFormat::WGSL). Specialization constants are WGSL overrides named by their id,
//   and a pipeline gets only those its shader declares: WebGPU refuses the rest, and a shader compiler drops what the
//   shader doesn't use.
// - A binding counts against the per-stage limits (16 storage buffers, 16 samplers...) in every stage its layout
//   entry is visible to, whether or not a shader there reads it. So an entry is visible only to the stages whose
//   shaders, of all created so far, declare that binding number (in any group), and a set's layout is rebuilt when a
//   new shader adds a stage to one of its bindings. A shader has to be created before the first draw that reads a
//   binding from a stage no other shader reads it from.
// - Depth textures are bound as unfilterable-float, which is how a shader written for the other backends reads them
//   (a texture of floats, not a depth texture).
// - Buffer copies are in whole 4-byte words: a copy whose offsets or size aren't is widened to the words around it,
//   so up to 3 bytes either side are copied too. That is exact for a staging buffer that mirrors its destination.
//
// The device is created by the page before the module starts (Module.preinitializedWebGPUDevice, see
// plume_webgpu_device.js), because requesting one is asynchronous. Emdawnwebgpu marks emscripten_webgpu_get_device(),
// which returns it, as deprecated; the replacement, requesting the adapter and device here and waiting with
// wgpuInstanceWaitAny(), needs ASYNCIFY or JSPI. RenderWindow is the CSS selector of the canvas, such as "#canvas".
//
// The examples build for the browser with emcmake and -DPLUME_BUILD_EXAMPLES=ON, given PLUME_DXC_EXECUTABLE (a dxc
// for the host) and naga-cli on the PATH or as PLUME_NAGA_EXECUTABLE; see examples/cmake/modules/PlumeWGSL.cmake.

#pragma once

#include "plume_render_interface.h"

#include <map>
#include <set>
#include <unordered_map>

#include <webgpu/webgpu.h>

namespace plume {
    static const uint32_t PushConstantsGroup = 3;
    static const uint32_t PushConstantsBinding = 999;

    struct WebGPUBuffer;
    struct WebGPUCommandQueue;
    struct WebGPUDevice;
    struct WebGPUGraphicsPipeline;
    struct WebGPUInterface;
    struct WebGPUPipelineLayout;
    struct WebGPUPool;
    struct WebGPUTexture;

    struct WebGPUBufferFormattedView : RenderBufferFormattedView {
        RenderFormat format = RenderFormat::UNKNOWN;
        WebGPUBuffer *buffer = nullptr;

        WebGPUBufferFormattedView(WebGPUBuffer *buffer, RenderFormat format);
        ~WebGPUBufferFormattedView() override;
    };

    // A READBACK buffer's asynchronous map, shared with the callback that finishes it, so a buffer destroyed while a
    // map is in flight is never touched.
    struct WebGPUReadback {
        WebGPUBuffer *buffer = nullptr;
        uint64_t serial = 0;
        bool pending = false;
    };

    struct WebGPUBuffer : RenderBuffer {
        WGPUBuffer buffer = nullptr;
        WebGPUDevice *device = nullptr;
        RenderBufferDesc desc;
        std::vector<uint8_t> shadow;
        std::shared_ptr<WebGPUReadback> readback;
        bool mapped = false;

        WebGPUBuffer(WebGPUDevice *device, WebGPUPool *pool, const RenderBufferDesc &desc);
        ~WebGPUBuffer() override;
        void *map(uint32_t subresource, const RenderRange *readRange) override;
        void unmap(uint32_t subresource, const RenderRange *writtenRange) override;
        std::unique_ptr<RenderBufferFormattedView> createBufferFormattedView(RenderFormat format) override;
        void setName(const std::string &name) override;
        uint64_t getDeviceAddress() const override;
        bool isMapReady() override;
        void flush(uint64_t begin, uint64_t end);
        void requestReadback();
        void waitForReadback();
        void cancelReadback();
    };

    struct WebGPUTexture : RenderTexture {
        WGPUTexture texture = nullptr;
        WGPUTextureView defaultView = nullptr;
        WebGPUDevice *device = nullptr;
        RenderTextureDesc desc;
        bool ownsTexture = true;

        WebGPUTexture() = default;
        WebGPUTexture(WebGPUDevice *device, WebGPUPool *pool, const RenderTextureDesc &desc);
        ~WebGPUTexture() override;
        std::unique_ptr<RenderTextureView> createTextureView(const RenderTextureViewDesc &desc) const override;
        void setName(const std::string &name) override;
        WGPUTextureView getDefaultView();
        void releaseHandle();
    };

    struct WebGPUTextureView : RenderTextureView {
        WGPUTextureView view = nullptr;
        const WebGPUTexture *texture = nullptr;
        RenderTextureViewDesc desc;

        WebGPUTextureView(const WebGPUTexture *texture, const RenderTextureViewDesc &desc);
        ~WebGPUTextureView() override;
    };

    struct WebGPUSampler : RenderSampler {
        WGPUSampler sampler = nullptr;
        RenderSamplerDesc desc;

        WebGPUSampler(WebGPUDevice *device, const RenderSamplerDesc &desc);
        ~WebGPUSampler() override;
    };

    struct WebGPUShader : RenderShader {
        WGPUShaderModule module = nullptr;
        std::string entryPointName;
        std::set<uint32_t> overrideIds;

        WebGPUShader(WebGPUDevice *device, const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format);
        ~WebGPUShader() override;
        void setName(const std::string &name) override;
    };

    struct WebGPUDescriptorSet : RenderDescriptorSet {
        struct Binding {
            RenderDescriptorRangeType type = RenderDescriptorRangeType::UNKNOWN;
            uint32_t binding = 0;
            const WebGPUBuffer *buffer = nullptr;
            uint64_t bufferOffset = 0;
            uint64_t bufferSize = 0;
            const WebGPUTexture *texture = nullptr;
            const WebGPUTextureView *textureView = nullptr;
            const WebGPUSampler *sampler = nullptr;
        };

        WebGPUDevice *device = nullptr;
        std::vector<Binding> bindings;
        WGPUBindGroupLayout layout = nullptr;
        WGPUBindGroup bindGroup = nullptr;
        std::vector<WGPUBindGroupLayoutEntry> layoutEntries;
        std::vector<WGPUBindGroupEntry> groupEntries;
        std::string layoutKey;
        WGPUBindGroupLayout layoutWithPushConstants = nullptr;
        std::map<std::pair<uint64_t, uint32_t>, WGPUBindGroup> groupsWithPushConstants;
        uint64_t bindingStagesRevision = 0;
        bool dirty = true;

        WebGPUDescriptorSet(WebGPUDevice *device, const RenderDescriptorSetDesc &desc);
        ~WebGPUDescriptorSet() override;
        void setBuffer(uint32_t descriptorIndex, const RenderBuffer *buffer, uint64_t bufferSize, const RenderBufferStructuredView *bufferStructuredView, const RenderBufferFormattedView *bufferFormattedView) override;
        void setTexture(uint32_t descriptorIndex, const RenderTexture *texture, RenderTextureLayout textureLayout, const RenderTextureView *textureView) override;
        void setSampler(uint32_t descriptorIndex, const RenderSampler *sampler) override;
        void setAccelerationStructure(uint32_t descriptorIndex, const RenderAccelerationStructure *accelerationStructure) override;
        void update();
        WGPUBindGroupLayout getLayoutWithPushConstants();
        WGPUBindGroup getBindGroupWithPushConstants(WGPUBuffer buffer, uint64_t chunkId, uint32_t bindingSize);
        void releaseGroupsWithPushConstants();
    };

    // A pipeline is created when it is first drawn with, after the RenderShader may be gone, so it keeps its own
    // reference to the module.
    struct WebGPUShaderStage {
        WGPUShaderModule module = nullptr;
        std::string entryPointName;
        std::set<uint32_t> overrideIds;

        WebGPUShaderStage() = default;
        WebGPUShaderStage(const RenderShader *shader);
        WebGPUShaderStage(const WebGPUShaderStage &) = delete;
        WebGPUShaderStage &operator=(const WebGPUShaderStage &) = delete;
        ~WebGPUShaderStage();
    };

    struct WebGPUPipeline : RenderPipeline {
        enum class Type {
            Unknown,
            Compute,
            Graphics
        };

        Type type = Type::Unknown;
        WebGPUDevice *device = nullptr;

        WebGPUPipeline(WebGPUDevice *device, Type type);
        virtual ~WebGPUPipeline() override;
        void setName(const std::string &name) override;
        RenderPipelineProgram getProgram(const std::string &name) const override;
    };

    struct WebGPUComputePipeline : WebGPUPipeline {
        RenderComputePipelineDesc desc;
        WebGPUShaderStage computeStage;
        std::vector<RenderSpecConstant> specConstants;
        std::map<WGPUPipelineLayout, WGPUComputePipeline> pipelines;

        WebGPUComputePipeline(WebGPUDevice *device, const RenderComputePipelineDesc &desc);
        ~WebGPUComputePipeline() override;
        WGPUComputePipeline get(WGPUPipelineLayout layout);
    };

    struct WebGPUGraphicsPipeline : WebGPUPipeline {
        RenderGraphicsPipelineDesc desc;
        WebGPUShaderStage vertexStage;
        WebGPUShaderStage pixelStage;
        std::vector<RenderInputSlot> inputSlots;
        std::vector<RenderInputElement> inputElements;
        std::vector<RenderSpecConstant> specConstants;
        std::map<WGPUPipelineLayout, WGPURenderPipeline> pipelines;

        WebGPUGraphicsPipeline(WebGPUDevice *device, const RenderGraphicsPipelineDesc &desc);
        ~WebGPUGraphicsPipeline() override;
        WGPURenderPipeline get(WGPUPipelineLayout layout);
    };

    struct WebGPUPipelineLayout : RenderPipelineLayout {
        WebGPUDevice *device = nullptr;
        uint32_t descriptorSetCount = 0;
        std::vector<RenderPushConstantRange> pushConstantRanges;
        uint32_t pushConstantsSize = 0;

        WebGPUPipelineLayout(WebGPUDevice *device, const RenderPipelineLayoutDesc &desc);
        ~WebGPUPipelineLayout() override;
    };

    struct WebGPUCommandFence : RenderCommandFence {
    };

    struct WebGPUCommandSemaphore : RenderCommandSemaphore {
    };

    struct WebGPUSwapChain : RenderSwapChain {
        WebGPUCommandQueue *commandQueue = nullptr;
        WGPUSurface surface = nullptr;
        std::string selector;
        RenderSwapChainDesc desc;
        std::vector<std::unique_ptr<WebGPUTexture>> textures;
        uint32_t nextTextureIndex = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        bool vsyncEnabled = true;
        double lastFrameTime = 0.0;
        std::vector<double> frameIntervals;
        uint32_t frameIntervalCursor = 0;

        WebGPUSwapChain(WebGPUCommandQueue *commandQueue, const RenderSwapChainDesc &desc);
        ~WebGPUSwapChain() override;
        bool present(uint32_t textureIndex, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount) override;
        void wait() override;
        bool resize() override;
        bool needsResize() const override;
        void setVsyncEnabled(bool vsyncEnabled) override;
        bool isVsyncEnabled() const override;
        uint32_t getWidth() const override;
        uint32_t getHeight() const override;
        RenderTexture *getTexture(uint32_t textureIndex) override;
        uint32_t getTextureCount() const override;
        bool acquireTexture(RenderCommandSemaphore *signalSemaphore, uint32_t *textureIndex) override;
        RenderWindow getWindow() const override;
        bool isEmpty() const override;
        uint32_t getRefreshRate() const override;
        void getCanvasSize(uint32_t &width, uint32_t &height) const;
    };

    struct WebGPUFramebuffer : RenderFramebuffer {
        std::vector<const WebGPUTexture *> colorTextures;
        std::vector<const WebGPUTextureView *> colorViews;
        const WebGPUTexture *depthTexture = nullptr;
        const WebGPUTextureView *depthView = nullptr;
        bool depthReadOnly = false;
        uint32_t width = 0;
        uint32_t height = 0;

        WebGPUFramebuffer(const RenderFramebufferDesc &desc);
        ~WebGPUFramebuffer() override;
        uint32_t getWidth() const override;
        uint32_t getHeight() const override;
    };

    struct WebGPUCommandList : RenderCommandList {
        static const uint32_t MaxDescriptorSets = 8;
        static const uint32_t MaxVertexBuffers = 8;
        static const uint32_t PushConstantsChunkSize = 64 * 1024;

        struct PushConstantsChunk {
            WGPUBuffer buffer = nullptr;
            uint64_t id = 0;
            std::map<uint32_t, WGPUBindGroup> bindGroups;
            std::vector<uint8_t> data;
            uint32_t used = 0;
        };

        struct BindState {
            const WebGPUPipelineLayout *layout = nullptr;
            WebGPUDescriptorSet *descriptorSets[MaxDescriptorSets] = {};
            std::vector<uint8_t> pushConstants;
            bool pushConstantsDirty = false;
        };

        WebGPUDevice *device = nullptr;
        WGPUCommandEncoder encoder = nullptr;
        WGPUCommandBuffer commandBuffer = nullptr;
        WGPURenderPassEncoder renderPass = nullptr;
        const WebGPUFramebuffer *framebuffer = nullptr;
        std::vector<std::pair<bool, RenderColor>> pendingColorClears;
        bool pendingDepthClear = false;
        bool pendingStencilClear = false;
        float pendingDepthValue = 1.0f;
        uint32_t pendingStencilValue = 0;
        const WebGPUPipeline *pipeline = nullptr;
        BindState graphics;
        BindState compute;
        RenderVertexBufferView vertexBuffers[MaxVertexBuffers] = {};
        uint32_t vertexBufferCount = 0;
        RenderIndexBufferView indexBuffer;
        RenderViewport viewport;
        RenderRect scissor;
        bool viewportSet = false;
        bool scissorSet = false;
        bool renderStateDirty = true;
        std::vector<PushConstantsChunk> pushConstantsChunks;
        uint32_t pushConstantsChunkIndex = 0;

        WebGPUCommandList(WebGPUDevice *device);
        ~WebGPUCommandList() override;
        void begin() override;
        void end() override;
        void barriers(RenderBarrierStages stages, const RenderBufferBarrier *bufferBarriers, uint32_t bufferBarriersCount, const RenderTextureBarrier *textureBarriers, uint32_t textureBarriersCount) override;
        void dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) override;
        void traceRays(uint32_t width, uint32_t height, uint32_t depth, RenderBufferReference shaderBindingTable, const RenderShaderBindingGroupsInfo &shaderBindingGroupsInfo) override;
        void drawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation) override;
        void drawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation) override;
        void setPipeline(const RenderPipeline *pipeline) override;
        void setComputePipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setComputePushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override;
        void setComputeDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setGraphicsPipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setGraphicsPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override;
        void setGraphicsDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setGraphicsRootDescriptor(RenderBufferReference bufferReference, uint32_t rootDescriptorIndex) override;
        void setRaytracingPipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setRaytracingPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override;
        void setRaytracingDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setIndexBuffer(const RenderIndexBufferView *view) override;
        void setVertexBuffers(uint32_t startSlot, const RenderVertexBufferView *views, uint32_t viewCount, const RenderInputSlot *inputSlots) override;
        void setViewports(const RenderViewport *viewports, uint32_t count) override;
        void setScissors(const RenderRect *scissorRects, uint32_t count) override;
        void setFramebuffer(const RenderFramebuffer *framebuffer) override;
        void setDepthBias(float depthBias, float depthBiasClamp, float slopeScaledDepthBias) override;
        void clearColor(uint32_t attachmentIndex, RenderColor colorValue, const RenderRect *clearRects, uint32_t clearRectsCount) override;
        void clearDepthStencil(bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *clearRects, uint32_t clearRectsCount) override;
        void copyBufferRegion(RenderBufferReference dstBuffer, RenderBufferReference srcBuffer, uint64_t size) override;
        void copyTextureRegion(const RenderTextureCopyLocation &dstLocation, const RenderTextureCopyLocation &srcLocation, uint32_t dstX, uint32_t dstY, uint32_t dstZ, const RenderBox *srcBox) override;
        void copyBuffer(const RenderBuffer *dstBuffer, const RenderBuffer *srcBuffer) override;
        void copyTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override;
        void resolveTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override;
        void resolveTextureRegion(const RenderTexture *dstTexture, uint32_t dstX, uint32_t dstY, const RenderTexture *srcTexture, const RenderRect *srcRect, RenderResolveMode resolveMode) override;
        void buildBottomLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, const RenderBottomLevelASBuildInfo &buildInfo) override;
        void buildTopLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, RenderBufferReference instancesBuffer, const RenderTopLevelASBuildInfo &buildInfo) override;
        void discardTexture(const RenderTexture *texture) override;
        void resetQueryPool(const RenderQueryPool *queryPool, uint32_t queryFirstIndex, uint32_t queryCount) override;
        void writeTimestamp(const RenderQueryPool *queryPool, uint32_t queryIndex) override;

        std::vector<WebGPUBuffer *> readbackDestinations;

        void endRenderPass();
        bool beginRenderPass();
        void clearRects(int32_t colorAttachment, const RenderColor &color, bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *rects, uint32_t rectCount);
        void copiedInto(const RenderBuffer *buffer);
        bool prepareDraw();
        void setPushConstants(BindState &state, uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size);
        WGPUPipelineLayout resolveBindings(BindState &state);
        void bindGroups(BindState &state, WGPURenderPassEncoder renderPass, WGPUComputePassEncoder computePass);
    };

    struct WebGPUCommandQueue : RenderCommandQueue {
        WebGPUDevice *device = nullptr;

        WebGPUCommandQueue(WebGPUDevice *device);
        ~WebGPUCommandQueue() override;
        std::unique_ptr<RenderCommandList> createCommandList() override;
        std::unique_ptr<RenderSwapChain> createSwapChain(const RenderSwapChainDesc &desc) override;
        void executeCommandLists(const RenderCommandList **commandLists, uint32_t commandListCount, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount, RenderCommandSemaphore **signalSemaphores, uint32_t signalSemaphoreCount, RenderCommandFence *signalFence) override;
        void waitForCommandFence(RenderCommandFence *fence) override;
    };

    struct WebGPUPool : RenderPool {
        WebGPUDevice *device = nullptr;
        RenderPoolDesc desc;

        WebGPUPool(WebGPUDevice *device, const RenderPoolDesc &desc);
        ~WebGPUPool() override;
        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override;
        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override;
    };

    struct WebGPUQueryPool : RenderQueryPool {
        std::vector<uint64_t> results;

        WebGPUQueryPool(uint32_t queryCount);
        ~WebGPUQueryPool() override;
        void queryResults() override;
        const uint64_t *getResults() const override;
        uint32_t getCount() const override;
    };

    struct WebGPUDevice : RenderDevice {
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        WebGPUInterface *renderInterface = nullptr;
        RenderDeviceCapabilities capabilities;
        RenderDeviceDescription description;
        bool float32Filterable = false;
        bool float32Blendable = false;
        uint64_t pushConstantsChunkCount = 0;
        bool depthClipControl = false;
        WGPUBindGroupLayout emptyLayout = nullptr;
        WGPUBindGroup emptyBindGroup = nullptr;
        WGPUBindGroupLayoutEntry pushConstantsEntry = {};
        WGPUBindGroupLayout pushConstantsLayout = nullptr;
        std::unordered_map<std::string, WGPUBindGroupLayout> bindGroupLayouts;
        std::unordered_map<uint32_t, WGPUShaderStage> bindingStages;
        uint64_t bindingStagesRevision = 0;
        std::map<std::vector<WGPUBindGroupLayout>, WGPUPipelineLayout> pipelineLayouts;
        std::vector<WebGPUBuffer *> mappedBuffers;
        WGPUShaderModule clearShader = nullptr;
        std::unordered_map<std::string, WGPURenderPipeline> clearPipelines;

        WebGPUDevice(WebGPUInterface *renderInterface);
        ~WebGPUDevice() override;
        std::unique_ptr<RenderDescriptorSet> createDescriptorSet(const RenderDescriptorSetDesc &desc) override;
        std::unique_ptr<RenderShader> createShader(const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format) override;
        std::unique_ptr<RenderSampler> createSampler(const RenderSamplerDesc &desc) override;
        std::unique_ptr<RenderPipeline> createComputePipeline(const RenderComputePipelineDesc &desc) override;
        std::unique_ptr<RenderPipeline> createGraphicsPipeline(const RenderGraphicsPipelineDesc &desc) override;
        std::unique_ptr<RenderPipeline> createRaytracingPipeline(const RenderRaytracingPipelineDesc &desc, const RenderPipeline *previousPipeline) override;
        std::unique_ptr<RenderCommandQueue> createCommandQueue(RenderCommandListType type) override;
        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override;
        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override;
        std::unique_ptr<RenderAccelerationStructure> createAccelerationStructure(const RenderAccelerationStructureDesc &desc) override;
        std::unique_ptr<RenderPool> createPool(const RenderPoolDesc &desc) override;
        std::unique_ptr<RenderPipelineLayout> createPipelineLayout(const RenderPipelineLayoutDesc &desc) override;
        std::unique_ptr<RenderCommandFence> createCommandFence() override;
        std::unique_ptr<RenderCommandSemaphore> createCommandSemaphore() override;
        std::unique_ptr<RenderFramebuffer> createFramebuffer(const RenderFramebufferDesc &desc) override;
        std::unique_ptr<RenderQueryPool> createQueryPool(uint32_t queryCount) override;
        void setBottomLevelASBuildInfo(RenderBottomLevelASBuildInfo &buildInfo, const RenderBottomLevelASMesh *meshes, uint32_t meshCount, bool preferFastBuild, bool preferFastTrace) override;
        void setTopLevelASBuildInfo(RenderTopLevelASBuildInfo &buildInfo, const RenderTopLevelASInstance *instances, uint32_t instanceCount, bool preferFastBuild, bool preferFastTrace) override;
        void setShaderBindingTableInfo(RenderShaderBindingTableInfo &tableInfo, const RenderShaderBindingGroups &groups, const RenderPipeline *pipeline, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount) override;
        const RenderDeviceCapabilities &getCapabilities() const override;
        const RenderDeviceDescription &getDescription() const override;
        RenderSampleCounts getSampleCountsSupported(RenderFormat format) const override;
        bool beginCapture() override;
        bool endCapture() override;
        WGPUBindGroupLayout getBindGroupLayout(const std::string &key, const std::vector<WGPUBindGroupLayoutEntry> &entries);
        WGPUPipelineLayout getPipelineLayout(const std::vector<WGPUBindGroupLayout> &layouts);
        void flushMappedBuffers();
        WGPURenderPipeline getClearPipeline(const WebGPUFramebuffer *framebuffer, int32_t colorAttachment, bool clearDepth, bool clearStencil);
    };

    struct WebGPUInterface : RenderInterface {
        RenderInterfaceCapabilities capabilities;
        std::vector<std::string> deviceNames;

        WebGPUInterface();
        ~WebGPUInterface() override;
        std::unique_ptr<RenderDevice> createDevice(const std::string &preferredDeviceName) override;
        const RenderInterfaceCapabilities &getCapabilities() const override;
        const std::vector<std::string> &getDeviceNames() const override;
        bool isValid() const;
    };

    extern std::unique_ptr<RenderInterface> CreateWebGPUInterface();
};
