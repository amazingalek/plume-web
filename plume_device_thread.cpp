//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

// See plume_device_thread.h for what runs where.

#include "plume_device_thread.h"

#include <algorithm>
#include <cstring>
#include <type_traits>

#if defined(__EMSCRIPTEN_PTHREADS__)
#include <emscripten/proxying.h>
#endif

namespace plume {
    // Runs f on the device thread and returns what it returned.
    template<typename F>
    static auto runSync(DeviceThread *thread, F &&f) -> decltype(f()) {
        using Result = decltype(f());
        if constexpr (std::is_void_v<Result>) {
            thread->runSync(f);
        }
        else {
            Result result{};
            thread->runSync([&]() { result = f(); });
            return result;
        }
    }

    // Deletes a backend object on the device thread.
    template<typename T>
    static void destroyAsync(DeviceThread *thread, T *object) {
        if (object != nullptr) {
            thread->runAsync([object]() { delete object; });
        }
    }

    struct DTDevice;
    struct DTCommandQueue;
    struct DTPipelineLayout;

    struct DTBufferFormattedView : RenderBufferFormattedView {
        DeviceThread *thread = nullptr;
        RenderBufferFormattedView *inner = nullptr;

        DTBufferFormattedView(DeviceThread *thread, std::unique_ptr<RenderBufferFormattedView> inner) : thread(thread), inner(inner.release()) { }
        ~DTBufferFormattedView() override { destroyAsync(thread, inner); }
    };

    struct DTBuffer : RenderBuffer {
        DeviceThread *thread = nullptr;
        RenderBuffer *inner = nullptr;

        DTBuffer(DeviceThread *thread, std::unique_ptr<RenderBuffer> inner) : thread(thread), inner(inner.release()) { }
        ~DTBuffer() override { destroyAsync(thread, inner); }

        void *map(uint32_t subresource, const RenderRange *readRange) override {
            return runSync(thread, [&]() { return inner->map(subresource, readRange); });
        }

        void unmap(uint32_t subresource, const RenderRange *writtenRange) override {
            RenderBuffer *buffer = inner;
            const bool ranged = (writtenRange != nullptr);
            const RenderRange range = ranged ? *writtenRange : RenderRange();
            thread->runAsync([=]() { buffer->unmap(subresource, ranged ? &range : nullptr); });
        }

        std::unique_ptr<RenderBufferFormattedView> createBufferFormattedView(RenderFormat format) override {
            std::unique_ptr<RenderBufferFormattedView> view = runSync(thread, [&]() { return inner->createBufferFormattedView(format); });
            return view ? std::make_unique<DTBufferFormattedView>(thread, std::move(view)) : nullptr;
        }

        void setName(const std::string &name) override {
            thread->runAsync([buffer = inner, name]() { buffer->setName(name); });
        }

        uint64_t getDeviceAddress() const override {
            return runSync(thread, [&]() { return inner->getDeviceAddress(); });
        }
    };

    struct DTTextureView : RenderTextureView {
        DeviceThread *thread = nullptr;
        RenderTextureView *inner = nullptr;

        DTTextureView(DeviceThread *thread, std::unique_ptr<RenderTextureView> inner) : thread(thread), inner(inner.release()) { }
        ~DTTextureView() override { destroyAsync(thread, inner); }
    };

    struct DTTexture : RenderTexture {
        DeviceThread *thread = nullptr;
        RenderTexture *inner = nullptr;
        bool ownsInner = true;

        DTTexture(DeviceThread *thread, std::unique_ptr<RenderTexture> inner) : thread(thread), inner(inner.release()) { }
        DTTexture(DeviceThread *thread) : thread(thread), ownsInner(false) { }

        ~DTTexture() override {
            if (ownsInner) {
                destroyAsync(thread, inner);
            }
        }

        std::unique_ptr<RenderTextureView> createTextureView(const RenderTextureViewDesc &desc) const override {
            std::unique_ptr<RenderTextureView> view = runSync(thread, [&]() { return inner->createTextureView(desc); });
            return view ? std::make_unique<DTTextureView>(thread, std::move(view)) : nullptr;
        }

        void setName(const std::string &name) override {
            thread->runAsync([texture = inner, name]() { texture->setName(name); });
        }
    };

    struct DTShader : RenderShader {
        DeviceThread *thread = nullptr;
        RenderShader *inner = nullptr;

        DTShader(DeviceThread *thread, std::unique_ptr<RenderShader> inner) : thread(thread), inner(inner.release()) { }
        ~DTShader() override { destroyAsync(thread, inner); }

        void setName(const std::string &name) override {
            thread->runAsync([shader = inner, name]() { shader->setName(name); });
        }
    };

    struct DTSampler : RenderSampler {
        DeviceThread *thread = nullptr;
        RenderSampler *inner = nullptr;

        DTSampler(DeviceThread *thread, std::unique_ptr<RenderSampler> inner) : thread(thread), inner(inner.release()) { }
        ~DTSampler() override { destroyAsync(thread, inner); }
    };

    struct DTPipeline : RenderPipeline {
        DeviceThread *thread = nullptr;
        RenderPipeline *inner = nullptr;

        DTPipeline(DeviceThread *thread, std::unique_ptr<RenderPipeline> inner) : thread(thread), inner(inner.release()) { }
        ~DTPipeline() override { destroyAsync(thread, inner); }

        void setName(const std::string &name) override {
            thread->runAsync([pipeline = inner, name]() { pipeline->setName(name); });
        }

        RenderPipelineProgram getProgram(const std::string &name) const override {
            return runSync(thread, [&]() { return inner->getProgram(name); });
        }
    };

    struct DTPipelineLayout : RenderPipelineLayout {
        DeviceThread *thread = nullptr;
        RenderPipelineLayout *inner = nullptr;

        // For the size of a push constant write that passes 0.
        std::vector<RenderPushConstantRange> pushConstantRanges;

        DTPipelineLayout(DeviceThread *thread, std::unique_ptr<RenderPipelineLayout> inner, const RenderPipelineLayoutDesc &desc) : thread(thread), inner(inner.release()) {
            pushConstantRanges.assign(desc.pushConstantRanges, desc.pushConstantRanges + desc.pushConstantRangesCount);
        }

        ~DTPipelineLayout() override { destroyAsync(thread, inner); }
    };

    struct DTCommandFence : RenderCommandFence {
        DeviceThread *thread = nullptr;
        RenderCommandFence *inner = nullptr;

        DTCommandFence(DeviceThread *thread, std::unique_ptr<RenderCommandFence> inner) : thread(thread), inner(inner.release()) { }
        ~DTCommandFence() override { destroyAsync(thread, inner); }
    };

    struct DTCommandSemaphore : RenderCommandSemaphore {
        DeviceThread *thread = nullptr;
        RenderCommandSemaphore *inner = nullptr;

        DTCommandSemaphore(DeviceThread *thread, std::unique_ptr<RenderCommandSemaphore> inner) : thread(thread), inner(inner.release()) { }
        ~DTCommandSemaphore() override { destroyAsync(thread, inner); }
    };

    struct DTFramebuffer : RenderFramebuffer {
        DeviceThread *thread = nullptr;
        RenderFramebuffer *inner = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;

        DTFramebuffer(DeviceThread *thread, std::unique_ptr<RenderFramebuffer> inner) : thread(thread), inner(inner.release()) {
            width = this->inner->getWidth();
            height = this->inner->getHeight();
        }

        ~DTFramebuffer() override { destroyAsync(thread, inner); }
        uint32_t getWidth() const override { return width; }
        uint32_t getHeight() const override { return height; }
    };

    struct DTQueryPool : RenderQueryPool {
        DeviceThread *thread = nullptr;
        RenderQueryPool *inner = nullptr;
        uint32_t count = 0;

        DTQueryPool(DeviceThread *thread, std::unique_ptr<RenderQueryPool> inner) : thread(thread), inner(inner.release()) {
            count = this->inner->getCount();
        }

        ~DTQueryPool() override { destroyAsync(thread, inner); }

        void queryResults() override {
            runSync(thread, [&]() { inner->queryResults(); });
        }

        const uint64_t *getResults() const override {
            return runSync(thread, [&]() { return inner->getResults(); });
        }

        uint32_t getCount() const override {
            return count;
        }
    };

    // The backend object behind a wrapper.

    template<typename Wrapper, typename T>
    static auto unwrapAs(T *object) -> decltype(static_cast<Wrapper *>(object)->inner) {
        return (object != nullptr) ? static_cast<Wrapper *>(object)->inner : nullptr;
    }

    template<typename Wrapper, typename T>
    static auto unwrapAs(const T *object) -> decltype(static_cast<const Wrapper *>(object)->inner) {
        return (object != nullptr) ? static_cast<const Wrapper *>(object)->inner : nullptr;
    }

    static RenderBuffer *unwrap(const RenderBuffer *object) { return unwrapAs<DTBuffer>(object); }
    static RenderBufferFormattedView *unwrap(const RenderBufferFormattedView *object) { return unwrapAs<DTBufferFormattedView>(object); }
    static RenderTexture *unwrap(const RenderTexture *object) { return unwrapAs<DTTexture>(object); }
    static RenderTextureView *unwrap(const RenderTextureView *object) { return unwrapAs<DTTextureView>(object); }
    static RenderShader *unwrap(const RenderShader *object) { return unwrapAs<DTShader>(object); }
    static RenderSampler *unwrap(const RenderSampler *object) { return unwrapAs<DTSampler>(object); }
    static RenderPipeline *unwrap(const RenderPipeline *object) { return unwrapAs<DTPipeline>(object); }
    static RenderPipelineLayout *unwrap(const RenderPipelineLayout *object) { return unwrapAs<DTPipelineLayout>(object); }
    static RenderCommandFence *unwrap(const RenderCommandFence *object) { return unwrapAs<DTCommandFence>(object); }
    static RenderCommandSemaphore *unwrap(const RenderCommandSemaphore *object) { return unwrapAs<DTCommandSemaphore>(object); }
    static RenderFramebuffer *unwrap(const RenderFramebuffer *object) { return unwrapAs<DTFramebuffer>(object); }
    static RenderQueryPool *unwrap(const RenderQueryPool *object) { return unwrapAs<DTQueryPool>(object); }

    static RenderBufferReference unwrap(const RenderBufferReference &reference) {
        return RenderBufferReference(unwrap(reference.ref), reference.offset);
    }

    static std::vector<RenderCommandSemaphore *> unwrap(RenderCommandSemaphore **semaphores, uint32_t count) {
        std::vector<RenderCommandSemaphore *> unwrapped(count);
        for (uint32_t i = 0; i < count; i++) {
            unwrapped[i] = unwrap(semaphores[i]);
        }

        return unwrapped;
    }

    // A descriptor set layout with its immutable samplers swapped for the backend's.
    struct UnwrappedDescriptorSetDesc {
        RenderDescriptorSetDesc desc;
        std::vector<RenderDescriptorRange> ranges;
        std::vector<std::vector<const RenderSampler *>> samplers;

        UnwrappedDescriptorSetDesc(const RenderDescriptorSetDesc &source) : desc(source) {
            ranges.assign(source.descriptorRanges, source.descriptorRanges + source.descriptorRangesCount);
            samplers.resize(ranges.size());
            for (size_t i = 0; i < ranges.size(); i++) {
                if (ranges[i].immutableSampler != nullptr) {
                    for (uint32_t j = 0; j < ranges[i].count; j++) {
                        samplers[i].emplace_back(unwrap(ranges[i].immutableSampler[j]));
                    }

                    ranges[i].immutableSampler = samplers[i].data();
                }
            }

            desc.descriptorRanges = ranges.data();
        }
    };

    struct DTDescriptorSet : RenderDescriptorSet {
        DeviceThread *thread = nullptr;
        RenderDescriptorSet *inner = nullptr;

        DTDescriptorSet(DeviceThread *thread, std::unique_ptr<RenderDescriptorSet> inner) : thread(thread), inner(inner.release()) { }
        ~DTDescriptorSet() override { destroyAsync(thread, inner); }

        void setBuffer(uint32_t descriptorIndex, const RenderBuffer *buffer, uint64_t bufferSize, const RenderBufferStructuredView *bufferStructuredView, const RenderBufferFormattedView *bufferFormattedView) override {
            const bool structured = (bufferStructuredView != nullptr);
            const RenderBufferStructuredView structuredView = structured ? *bufferStructuredView : RenderBufferStructuredView();
            thread->runAsync([set = inner, descriptorIndex, innerBuffer = unwrap(buffer), bufferSize, structured, structuredView, formattedView = unwrap(bufferFormattedView)]() {
                set->setBuffer(descriptorIndex, innerBuffer, bufferSize, structured ? &structuredView : nullptr, formattedView);
            });
        }

        void setTexture(uint32_t descriptorIndex, const RenderTexture *texture, RenderTextureLayout textureLayout, const RenderTextureView *textureView) override {
            thread->runAsync([set = inner, descriptorIndex, innerTexture = unwrap(texture), textureLayout, innerView = unwrap(textureView)]() {
                set->setTexture(descriptorIndex, innerTexture, textureLayout, innerView);
            });
        }

        void setSampler(uint32_t descriptorIndex, const RenderSampler *sampler) override {
            thread->runAsync([set = inner, descriptorIndex, innerSampler = unwrap(sampler)]() {
                set->setSampler(descriptorIndex, innerSampler);
            });
        }

        void setAccelerationStructure(uint32_t descriptorIndex, const RenderAccelerationStructure *accelerationStructure) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }
    };

    static RenderDescriptorSet *unwrap(const RenderDescriptorSet *object) { return unwrapAs<DTDescriptorSet>(object); }

    // DTCommandList

    struct DTCommandList : RenderCommandList {
        typedef std::function<void(RenderCommandList *)> Command;
        typedef std::vector<Command> Recording;

        DeviceThread *thread = nullptr;
        RenderCommandList *inner = nullptr;
        std::shared_ptr<Recording> recording;
        std::shared_ptr<const Recording> recorded;
        const DTPipelineLayout *computeLayout = nullptr;
        const DTPipelineLayout *graphicsLayout = nullptr;

        DTCommandList(DeviceThread *thread, std::unique_ptr<RenderCommandList> inner) : thread(thread), inner(inner.release()) { }
        ~DTCommandList() override { destroyAsync(thread, inner); }

        void record(Command &&command) {
            assert((recording != nullptr) && "The command list wasn't begun.");
            recording->emplace_back(std::move(command));
        }

        // Replays the last recording into the backend's command list. On the device thread.
        static void replay(RenderCommandList *commandList, const Recording &recording) {
            commandList->begin();
            for (const Command &command : recording) {
                command(commandList);
            }

            commandList->end();
        }

        static std::vector<uint8_t> copyPushConstants(const DTPipelineLayout *layout, uint32_t rangeIndex, const void *data, uint32_t size) {
            if (size == 0) {
                assert((layout != nullptr) && (rangeIndex < layout->pushConstantRanges.size()) && "Push constants without a size need the pipeline layout set first.");
                size = layout->pushConstantRanges[rangeIndex].size;
            }

            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
            return std::vector<uint8_t>(bytes, bytes + size);
        }

        void begin() override {
            recording = std::make_shared<Recording>();
            computeLayout = nullptr;
            graphicsLayout = nullptr;
        }

        void end() override {
            recorded = std::move(recording);
            recording.reset();
        }

        void barriers(RenderBarrierStages stages, const RenderBufferBarrier *bufferBarriers, uint32_t bufferBarriersCount, const RenderTextureBarrier *textureBarriers, uint32_t textureBarriersCount) override {
            std::vector<RenderBufferBarrier> buffers(bufferBarriers, bufferBarriers + bufferBarriersCount);
            for (RenderBufferBarrier &barrier : buffers) {
                barrier.buffer = unwrap(barrier.buffer);
            }

            std::vector<RenderTextureBarrier> textures(textureBarriers, textureBarriers + textureBarriersCount);
            for (RenderTextureBarrier &barrier : textures) {
                barrier.texture = unwrap(barrier.texture);
            }

            record([=](RenderCommandList *c) { c->barriers(stages, buffers.data(), uint32_t(buffers.size()), textures.data(), uint32_t(textures.size())); });
        }

        void dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) override {
            record([=](RenderCommandList *c) { c->dispatch(threadGroupCountX, threadGroupCountY, threadGroupCountZ); });
        }

        void traceRays(uint32_t width, uint32_t height, uint32_t depth, RenderBufferReference shaderBindingTable, const RenderShaderBindingGroupsInfo &shaderBindingGroupsInfo) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        void drawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation) override {
            record([=](RenderCommandList *c) { c->drawInstanced(vertexCountPerInstance, instanceCount, startVertexLocation, startInstanceLocation); });
        }

        void drawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation) override {
            record([=](RenderCommandList *c) { c->drawIndexedInstanced(indexCountPerInstance, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation); });
        }

        void setPipeline(const RenderPipeline *pipeline) override {
            record([p = unwrap(pipeline)](RenderCommandList *c) { c->setPipeline(p); });
        }

        void setComputePipelineLayout(const RenderPipelineLayout *pipelineLayout) override {
            computeLayout = static_cast<const DTPipelineLayout *>(pipelineLayout);
            record([p = unwrap(pipelineLayout)](RenderCommandList *c) { c->setComputePipelineLayout(p); });
        }

        void setComputePushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override {
            std::vector<uint8_t> bytes = copyPushConstants(computeLayout, rangeIndex, data, size);
            record([=](RenderCommandList *c) { c->setComputePushConstants(rangeIndex, bytes.data(), offset, uint32_t(bytes.size())); });
        }

        void setComputeDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override {
            record([s = unwrap(descriptorSet), setIndex](RenderCommandList *c) { c->setComputeDescriptorSet(s, setIndex); });
        }

        void setGraphicsPipelineLayout(const RenderPipelineLayout *pipelineLayout) override {
            graphicsLayout = static_cast<const DTPipelineLayout *>(pipelineLayout);
            record([p = unwrap(pipelineLayout)](RenderCommandList *c) { c->setGraphicsPipelineLayout(p); });
        }

        void setGraphicsPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override {
            std::vector<uint8_t> bytes = copyPushConstants(graphicsLayout, rangeIndex, data, size);
            record([=](RenderCommandList *c) { c->setGraphicsPushConstants(rangeIndex, bytes.data(), offset, uint32_t(bytes.size())); });
        }

        void setGraphicsDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override {
            record([s = unwrap(descriptorSet), setIndex](RenderCommandList *c) { c->setGraphicsDescriptorSet(s, setIndex); });
        }

        void setGraphicsRootDescriptor(RenderBufferReference bufferReference, uint32_t rootDescriptorIndex) override {
            record([r = unwrap(bufferReference), rootDescriptorIndex](RenderCommandList *c) { c->setGraphicsRootDescriptor(r, rootDescriptorIndex); });
        }

        void setRaytracingPipelineLayout(const RenderPipelineLayout *pipelineLayout) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        void setRaytracingPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        void setRaytracingDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        void setIndexBuffer(const RenderIndexBufferView *view) override {
            const bool hasView = (view != nullptr);
            RenderIndexBufferView unwrapped = hasView ? *view : RenderIndexBufferView();
            unwrapped.buffer = unwrap(unwrapped.buffer);
            record([=](RenderCommandList *c) { c->setIndexBuffer(hasView ? &unwrapped : nullptr); });
        }

        void setVertexBuffers(uint32_t startSlot, const RenderVertexBufferView *views, uint32_t viewCount, const RenderInputSlot *inputSlots) override {
            std::vector<RenderVertexBufferView> unwrapped(views, views + ((views != nullptr) ? viewCount : 0));
            for (RenderVertexBufferView &view : unwrapped) {
                view.buffer = unwrap(view.buffer);
            }

            std::vector<RenderInputSlot> slots;
            if (inputSlots != nullptr) {
                slots.assign(inputSlots, inputSlots + viewCount);
            }

            record([=](RenderCommandList *c) { c->setVertexBuffers(startSlot, unwrapped.empty() ? nullptr : unwrapped.data(), viewCount, slots.empty() ? nullptr : slots.data()); });
        }

        void setViewports(const RenderViewport *viewports, uint32_t count) override {
            std::vector<RenderViewport> copied(viewports, viewports + count);
            record([=](RenderCommandList *c) { c->setViewports(copied.data(), uint32_t(copied.size())); });
        }

        void setScissors(const RenderRect *scissorRects, uint32_t count) override {
            std::vector<RenderRect> copied(scissorRects, scissorRects + count);
            record([=](RenderCommandList *c) { c->setScissors(copied.data(), uint32_t(copied.size())); });
        }

        void setFramebuffer(const RenderFramebuffer *framebuffer) override {
            record([f = unwrap(framebuffer)](RenderCommandList *c) { c->setFramebuffer(f); });
        }

        void setDepthBias(float depthBias, float depthBiasClamp, float slopeScaledDepthBias) override {
            record([=](RenderCommandList *c) { c->setDepthBias(depthBias, depthBiasClamp, slopeScaledDepthBias); });
        }

        void clearColor(uint32_t attachmentIndex, RenderColor colorValue, const RenderRect *clearRects, uint32_t clearRectsCount) override {
            std::vector<RenderRect> rects(clearRects, clearRects + ((clearRects != nullptr) ? clearRectsCount : 0));
            record([=](RenderCommandList *c) { c->clearColor(attachmentIndex, colorValue, rects.empty() ? nullptr : rects.data(), uint32_t(rects.size())); });
        }

        void clearDepthStencil(bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *clearRects, uint32_t clearRectsCount) override {
            std::vector<RenderRect> rects(clearRects, clearRects + ((clearRects != nullptr) ? clearRectsCount : 0));
            record([=](RenderCommandList *c) { c->clearDepthStencil(clearDepth, clearStencil, depthValue, stencilValue, rects.empty() ? nullptr : rects.data(), uint32_t(rects.size())); });
        }

        void copyBufferRegion(RenderBufferReference dstBuffer, RenderBufferReference srcBuffer, uint64_t size) override {
            record([dst = unwrap(dstBuffer), src = unwrap(srcBuffer), size](RenderCommandList *c) { c->copyBufferRegion(dst, src, size); });
        }

        void copyTextureRegion(const RenderTextureCopyLocation &dstLocation, const RenderTextureCopyLocation &srcLocation, uint32_t dstX, uint32_t dstY, uint32_t dstZ, const RenderBox *srcBox) override {
            RenderTextureCopyLocation dst = dstLocation;
            dst.texture = unwrap(dst.texture);
            dst.buffer = unwrap(dst.buffer);
            RenderTextureCopyLocation src = srcLocation;
            src.texture = unwrap(src.texture);
            src.buffer = unwrap(src.buffer);
            const bool hasBox = (srcBox != nullptr);
            const RenderBox box = hasBox ? *srcBox : RenderBox();
            record([=](RenderCommandList *c) { c->copyTextureRegion(dst, src, dstX, dstY, dstZ, hasBox ? &box : nullptr); });
        }

        void copyBuffer(const RenderBuffer *dstBuffer, const RenderBuffer *srcBuffer) override {
            record([dst = unwrap(dstBuffer), src = unwrap(srcBuffer)](RenderCommandList *c) { c->copyBuffer(dst, src); });
        }

        void copyTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override {
            record([dst = unwrap(dstTexture), src = unwrap(srcTexture)](RenderCommandList *c) { c->copyTexture(dst, src); });
        }

        void resolveTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override {
            record([dst = unwrap(dstTexture), src = unwrap(srcTexture)](RenderCommandList *c) { c->resolveTexture(dst, src); });
        }

        void resolveTextureRegion(const RenderTexture *dstTexture, uint32_t dstX, uint32_t dstY, const RenderTexture *srcTexture, const RenderRect *srcRect, RenderResolveMode resolveMode) override {
            const bool hasRect = (srcRect != nullptr);
            const RenderRect rect = hasRect ? *srcRect : RenderRect();
            record([=, dst = unwrap(dstTexture), src = unwrap(srcTexture)](RenderCommandList *c) { c->resolveTextureRegion(dst, dstX, dstY, src, hasRect ? &rect : nullptr, resolveMode); });
        }

        void buildBottomLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, const RenderBottomLevelASBuildInfo &buildInfo) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }

        void buildTopLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, RenderBufferReference instancesBuffer, const RenderTopLevelASBuildInfo &buildInfo) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }

        void discardTexture(const RenderTexture *texture) override {
            record([t = unwrap(texture)](RenderCommandList *c) { c->discardTexture(t); });
        }

        void resetQueryPool(const RenderQueryPool *queryPool, uint32_t queryFirstIndex, uint32_t queryCount) override {
            record([q = unwrap(queryPool), queryFirstIndex, queryCount](RenderCommandList *c) { c->resetQueryPool(q, queryFirstIndex, queryCount); });
        }

        void writeTimestamp(const RenderQueryPool *queryPool, uint32_t queryIndex) override {
            record([q = unwrap(queryPool), queryIndex](RenderCommandList *c) { c->writeTimestamp(q, queryIndex); });
        }
    };

    // DTSwapChain

    struct DTSwapChain : RenderSwapChain {
        DeviceThread *thread = nullptr;
        RenderDevice *innerDevice = nullptr;
        RenderCommandQueue *innerQueue = nullptr;
        RenderSwapChain *inner = nullptr;
        RenderSwapChainDesc desc;
        RenderWindow window = {};

        // What the caller draws into, and what present() copies to the backend's swap chain texture with.
        std::vector<std::unique_ptr<DTTexture>> textures;
        std::vector<std::unique_ptr<RenderTexture>> innerTextures;
        std::unique_ptr<RenderCommandList> copyList;
        std::unique_ptr<RenderCommandSemaphore> acquireSemaphore;
        std::unique_ptr<RenderCommandSemaphore> copySemaphore;

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t nextTextureIndex = 0;
        bool vsyncEnabled = true;

        // On the device thread.
        DTSwapChain(DeviceThread *thread, RenderDevice *innerDevice, RenderCommandQueue *innerQueue, std::unique_ptr<RenderSwapChain> inner, const RenderSwapChainDesc &desc)
            : thread(thread), innerDevice(innerDevice), innerQueue(innerQueue), inner(inner.release()), desc(desc)
        {
            window = this->inner->getWindow();
            vsyncEnabled = this->inner->isVsyncEnabled();
            copyList = innerQueue->createCommandList();
            acquireSemaphore = innerDevice->createCommandSemaphore();
            copySemaphore = innerDevice->createCommandSemaphore();
            for (uint32_t i = 0; i < std::max(desc.textureCount, 1U); i++) {
                textures.emplace_back(std::make_unique<DTTexture>(thread));
            }

            innerTextures.resize(textures.size());
            updateTextures();
        }

        ~DTSwapChain() override {
            thread->runSync([&]() {
                textures.clear();
                innerTextures.clear();
                copyList.reset();
                acquireSemaphore.reset();
                copySemaphore.reset();
                delete inner;
            });
        }

        // On the device thread: makes the caller's textures the size of the backend's swap chain.
        void updateTextures() {
            const uint32_t newWidth = inner->getWidth();
            const uint32_t newHeight = inner->getHeight();
            if ((newWidth == width) && (newHeight == height)) {
                return;
            }

            width = newWidth;
            height = newHeight;
            for (size_t i = 0; i < textures.size(); i++) {
                innerTextures[i].reset();
                if (!isEmpty()) {
                    innerTextures[i] = innerDevice->createTexture(RenderTextureDesc::ColorTarget(width, height, desc.format));
                }

                textures[i]->inner = innerTextures[i].get();
            }
        }

        bool present(uint32_t textureIndex, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount) override {
            assert(textureIndex < textures.size());
            std::vector<RenderCommandSemaphore *> waits = unwrap(waitSemaphores, waitSemaphoreCount);
            thread->runAsync([this, textureIndex, waits]() mutable {
                RenderTexture *source = innerTextures[textureIndex].get();
                uint32_t innerIndex = 0;
                if ((source == nullptr) || !inner->acquireTexture(acquireSemaphore.get(), &innerIndex)) {
                    return;
                }

                RenderTexture *target = inner->getTexture(innerIndex);
                copyList->begin();
                copyList->barriers(RenderBarrierStage::COPY, { RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE), RenderTextureBarrier(target, RenderTextureLayout::COPY_DEST) });
                copyList->copyTexture(target, source);
                copyList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(target, RenderTextureLayout::PRESENT));
                copyList->end();

                waits.emplace_back(acquireSemaphore.get());
                RenderCommandSemaphore *signal = copySemaphore.get();
                const RenderCommandList *commandList = copyList.get();
                innerQueue->executeCommandLists(&commandList, 1, waits.data(), uint32_t(waits.size()), &signal, 1);
                inner->present(innerIndex, &signal, 1);
            });

            return true;
        }

        void wait() override {
            runSync(thread, [&]() { inner->wait(); });
        }

        bool resize() override {
            return runSync(thread, [&]() {
                const bool resized = inner->resize();
                updateTextures();
                return resized;
            });
        }

        bool needsResize() const override {
            return runSync(thread, [&]() { return inner->needsResize(); });
        }

        void setVsyncEnabled(bool vsyncEnabled) override {
            this->vsyncEnabled = vsyncEnabled;
            thread->runAsync([swapChain = inner, vsyncEnabled]() { swapChain->setVsyncEnabled(vsyncEnabled); });
        }

        bool isVsyncEnabled() const override {
            return vsyncEnabled;
        }

        uint32_t getWidth() const override {
            return width;
        }

        uint32_t getHeight() const override {
            return height;
        }

        RenderTexture *getTexture(uint32_t textureIndex) override {
            assert(textureIndex < textures.size());
            return textures[textureIndex].get();
        }

        uint32_t getTextureCount() const override {
            return uint32_t(textures.size());
        }

        bool acquireTexture(RenderCommandSemaphore *signalSemaphore, uint32_t *textureIndex) override {
            assert(textureIndex != nullptr);
            if (isEmpty()) {
                return false;
            }

            *textureIndex = nextTextureIndex;
            nextTextureIndex = (nextTextureIndex + 1) % textures.size();

            // Nothing waits for a texture of this swap chain, but the caller will wait for the semaphore.
            if (signalSemaphore != nullptr) {
                thread->runAsync([queue = innerQueue, signal = unwrap(signalSemaphore)]() mutable {
                    queue->executeCommandLists(nullptr, 0, nullptr, 0, &signal, 1);
                });
            }

            return true;
        }

        RenderWindow getWindow() const override {
            return window;
        }

        bool isEmpty() const override {
            return (width == 0) || (height == 0);
        }

        uint32_t getRefreshRate() const override {
            return runSync(thread, [&]() { return inner->getRefreshRate(); });
        }
    };

    // DTCommandQueue

    struct DTCommandQueue : RenderCommandQueue {
        DeviceThread *thread = nullptr;
        RenderDevice *innerDevice = nullptr;
        RenderCommandQueue *inner = nullptr;

        DTCommandQueue(DeviceThread *thread, RenderDevice *innerDevice, std::unique_ptr<RenderCommandQueue> inner) : thread(thread), innerDevice(innerDevice), inner(inner.release()) { }
        ~DTCommandQueue() override { destroyAsync(thread, inner); }

        std::unique_ptr<RenderCommandList> createCommandList() override {
            std::unique_ptr<RenderCommandList> commandList = runSync(thread, [&]() { return inner->createCommandList(); });
            return commandList ? std::make_unique<DTCommandList>(thread, std::move(commandList)) : nullptr;
        }

        std::unique_ptr<RenderSwapChain> createSwapChain(const RenderSwapChainDesc &desc) override {
            return runSync(thread, [&]() -> std::unique_ptr<RenderSwapChain> {
                std::unique_ptr<RenderSwapChain> swapChain = inner->createSwapChain(desc);
                if (swapChain == nullptr) {
                    return nullptr;
                }

                return std::make_unique<DTSwapChain>(thread, innerDevice, inner, std::move(swapChain), desc);
            });
        }

        void executeCommandLists(const RenderCommandList **commandLists, uint32_t commandListCount, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount, RenderCommandSemaphore **signalSemaphores, uint32_t signalSemaphoreCount, RenderCommandFence *signalFence) override {
            struct Execution {
                RenderCommandList *commandList;
                std::shared_ptr<const DTCommandList::Recording> recording;
            };

            std::vector<Execution> executions;
            for (uint32_t i = 0; i < commandListCount; i++) {
                const DTCommandList *commandList = static_cast<const DTCommandList *>(commandLists[i]);
                assert((commandList->recorded != nullptr) && "The command list wasn't ended.");
                executions.push_back({ commandList->inner, commandList->recorded });
            }

            thread->runAsync([queue = inner, executions = std::move(executions), waits = unwrap(waitSemaphores, waitSemaphoreCount), signals = unwrap(signalSemaphores, signalSemaphoreCount), fence = unwrap(signalFence)]() mutable {
                std::vector<const RenderCommandList *> innerLists;
                for (const Execution &execution : executions) {
                    DTCommandList::replay(execution.commandList, *execution.recording);
                    innerLists.emplace_back(execution.commandList);
                }

                queue->executeCommandLists(innerLists.data(), uint32_t(innerLists.size()), waits.data(), uint32_t(waits.size()), signals.data(), uint32_t(signals.size()), fence);
            });
        }

        void waitForCommandFence(RenderCommandFence *fence) override {
            runSync(thread, [&]() { inner->waitForCommandFence(unwrap(fence)); });
        }
    };

    // DTPool

    struct DTPool : RenderPool {
        DeviceThread *thread = nullptr;
        RenderPool *inner = nullptr;

        DTPool(DeviceThread *thread, std::unique_ptr<RenderPool> inner) : thread(thread), inner(inner.release()) { }
        ~DTPool() override { destroyAsync(thread, inner); }

        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override {
            std::unique_ptr<RenderBuffer> buffer = runSync(thread, [&]() { return inner->createBuffer(desc); });
            return buffer ? std::make_unique<DTBuffer>(thread, std::move(buffer)) : nullptr;
        }

        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override {
            std::unique_ptr<RenderTexture> texture = runSync(thread, [&]() { return inner->createTexture(desc); });
            return texture ? std::make_unique<DTTexture>(thread, std::move(texture)) : nullptr;
        }
    };

    // DTDevice

    struct DTDevice : RenderDevice {
        DeviceThread *thread = nullptr;
        RenderDevice *inner = nullptr;
        RenderDeviceCapabilities capabilities;
        RenderDeviceDescription description;

        DTDevice(DeviceThread *thread, std::unique_ptr<RenderDevice> inner) : thread(thread), inner(inner.release()) {
            capabilities = this->inner->getCapabilities();
            description = this->inner->getDescription();
        }

        ~DTDevice() override { destroyAsync(thread, inner); }

        template<typename Wrapper, typename T, typename F>
        std::unique_ptr<T> create(F &&f) {
            std::unique_ptr<T> object = runSync(thread, f);
            return object ? std::make_unique<Wrapper>(thread, std::move(object)) : nullptr;
        }

        std::unique_ptr<RenderDescriptorSet> createDescriptorSet(const RenderDescriptorSetDesc &desc) override {
            return create<DTDescriptorSet, RenderDescriptorSet>([&]() {
                UnwrappedDescriptorSetDesc unwrapped(desc);
                return inner->createDescriptorSet(unwrapped.desc);
            });
        }

        std::unique_ptr<RenderShader> createShader(const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format) override {
            return create<DTShader, RenderShader>([&]() { return inner->createShader(data, size, entryPointName, format); });
        }

        std::unique_ptr<RenderSampler> createSampler(const RenderSamplerDesc &desc) override {
            return create<DTSampler, RenderSampler>([&]() { return inner->createSampler(desc); });
        }

        std::unique_ptr<RenderPipeline> createComputePipeline(const RenderComputePipelineDesc &desc) override {
            RenderComputePipelineDesc unwrapped = desc;
            unwrapped.pipelineLayout = unwrap(desc.pipelineLayout);
            unwrapped.computeShader = unwrap(desc.computeShader);
            return create<DTPipeline, RenderPipeline>([&]() { return inner->createComputePipeline(unwrapped); });
        }

        std::unique_ptr<RenderPipeline> createGraphicsPipeline(const RenderGraphicsPipelineDesc &desc) override {
            RenderGraphicsPipelineDesc unwrapped = desc;
            unwrapped.pipelineLayout = unwrap(desc.pipelineLayout);
            unwrapped.vertexShader = unwrap(desc.vertexShader);
            unwrapped.geometryShader = unwrap(desc.geometryShader);
            unwrapped.pixelShader = unwrap(desc.pixelShader);
            return create<DTPipeline, RenderPipeline>([&]() { return inner->createGraphicsPipeline(unwrapped); });
        }

        std::unique_ptr<RenderPipeline> createRaytracingPipeline(const RenderRaytracingPipelineDesc &desc, const RenderPipeline *previousPipeline) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
            return nullptr;
        }

        std::unique_ptr<RenderCommandQueue> createCommandQueue(RenderCommandListType type) override {
            std::unique_ptr<RenderCommandQueue> queue = runSync(thread, [&]() { return inner->createCommandQueue(type); });
            return queue ? std::make_unique<DTCommandQueue>(thread, inner, std::move(queue)) : nullptr;
        }

        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override {
            return create<DTBuffer, RenderBuffer>([&]() { return inner->createBuffer(desc); });
        }

        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override {
            return create<DTTexture, RenderTexture>([&]() { return inner->createTexture(desc); });
        }

        std::unique_ptr<RenderAccelerationStructure> createAccelerationStructure(const RenderAccelerationStructureDesc &desc) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
            return nullptr;
        }

        std::unique_ptr<RenderPool> createPool(const RenderPoolDesc &desc) override {
            return create<DTPool, RenderPool>([&]() { return inner->createPool(desc); });
        }

        std::unique_ptr<RenderPipelineLayout> createPipelineLayout(const RenderPipelineLayoutDesc &desc) override {
            std::unique_ptr<RenderPipelineLayout> layout = runSync(thread, [&]() {
                std::vector<UnwrappedDescriptorSetDesc> sets(desc.descriptorSetDescs, desc.descriptorSetDescs + desc.descriptorSetDescsCount);
                std::vector<RenderDescriptorSetDesc> setDescs;
                for (const UnwrappedDescriptorSetDesc &set : sets) {
                    setDescs.emplace_back(set.desc);
                }

                RenderPipelineLayoutDesc unwrapped = desc;
                unwrapped.descriptorSetDescs = setDescs.empty() ? nullptr : setDescs.data();
                return inner->createPipelineLayout(unwrapped);
            });

            return layout ? std::make_unique<DTPipelineLayout>(thread, std::move(layout), desc) : nullptr;
        }

        std::unique_ptr<RenderCommandFence> createCommandFence() override {
            return create<DTCommandFence, RenderCommandFence>([&]() { return inner->createCommandFence(); });
        }

        std::unique_ptr<RenderCommandSemaphore> createCommandSemaphore() override {
            return create<DTCommandSemaphore, RenderCommandSemaphore>([&]() { return inner->createCommandSemaphore(); });
        }

        std::unique_ptr<RenderFramebuffer> createFramebuffer(const RenderFramebufferDesc &desc) override {
            std::vector<const RenderTexture *> colorAttachments(desc.colorAttachmentsCount);
            std::vector<const RenderTextureView *> colorAttachmentViews(desc.colorAttachmentsCount);
            for (uint32_t i = 0; i < desc.colorAttachmentsCount; i++) {
                colorAttachments[i] = unwrap(desc.colorAttachments[i]);
                colorAttachmentViews[i] = (desc.colorAttachmentViews != nullptr) ? unwrap(desc.colorAttachmentViews[i]) : nullptr;
            }

            RenderFramebufferDesc unwrapped = desc;
            unwrapped.colorAttachments = colorAttachments.empty() ? nullptr : colorAttachments.data();
            unwrapped.colorAttachmentViews = (desc.colorAttachmentViews != nullptr) ? colorAttachmentViews.data() : nullptr;
            unwrapped.depthAttachment = unwrap(desc.depthAttachment);
            unwrapped.depthAttachmentView = unwrap(desc.depthAttachmentView);

            // Made on the device thread: the wrapper reads the framebuffer's size.
            return runSync(thread, [&]() -> std::unique_ptr<RenderFramebuffer> {
                std::unique_ptr<RenderFramebuffer> framebuffer = inner->createFramebuffer(unwrapped);
                return framebuffer ? std::make_unique<DTFramebuffer>(thread, std::move(framebuffer)) : nullptr;
            });
        }

        std::unique_ptr<RenderQueryPool> createQueryPool(uint32_t queryCount) override {
            return runSync(thread, [&]() -> std::unique_ptr<RenderQueryPool> {
                std::unique_ptr<RenderQueryPool> queryPool = inner->createQueryPool(queryCount);
                return queryPool ? std::make_unique<DTQueryPool>(thread, std::move(queryPool)) : nullptr;
            });
        }

        void setBottomLevelASBuildInfo(RenderBottomLevelASBuildInfo &buildInfo, const RenderBottomLevelASMesh *meshes, uint32_t meshCount, bool preferFastBuild, bool preferFastTrace) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }

        void setTopLevelASBuildInfo(RenderTopLevelASBuildInfo &buildInfo, const RenderTopLevelASInstance *instances, uint32_t instanceCount, bool preferFastBuild, bool preferFastTrace) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }

        void setShaderBindingTableInfo(RenderShaderBindingTableInfo &tableInfo, const RenderShaderBindingGroups &groups, const RenderPipeline *pipeline, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        const RenderDeviceCapabilities &getCapabilities() const override {
            return capabilities;
        }

        const RenderDeviceDescription &getDescription() const override {
            return description;
        }

        RenderSampleCounts getSampleCountsSupported(RenderFormat format) const override {
            return runSync(thread, [&]() { return inner->getSampleCountsSupported(format); });
        }

        bool beginCapture() override {
            return runSync(thread, [&]() { return inner->beginCapture(); });
        }

        bool endCapture() override {
            return runSync(thread, [&]() { return inner->endCapture(); });
        }
    };

    // DTInterface

    struct DTInterface : RenderInterface {
        std::shared_ptr<DeviceThread> thread;
        RenderInterface *inner = nullptr;
        std::vector<std::string> deviceNames;
        RenderInterfaceCapabilities capabilities;

        DTInterface(std::shared_ptr<DeviceThread> thread, std::unique_ptr<RenderInterface> inner) : thread(thread), inner(inner.release()) {
            deviceNames = this->inner->getDeviceNames();
            capabilities = this->inner->getCapabilities();
        }

        ~DTInterface() override {
            // Waits, so that nothing queued outlives the thread.
            thread->runSync([&]() { delete inner; });
        }

        std::unique_ptr<RenderDevice> createDevice(const std::string &preferredDeviceName) override {
            return runSync(thread.get(), [&]() -> std::unique_ptr<RenderDevice> {
                std::unique_ptr<RenderDevice> device = inner->createDevice(preferredDeviceName);
                return device ? std::make_unique<DTDevice>(thread.get(), std::move(device)) : nullptr;
            });
        }

        const std::vector<std::string> &getDeviceNames() const override {
            return deviceNames;
        }

        const RenderInterfaceCapabilities &getCapabilities() const override {
            return capabilities;
        }
    };

    std::unique_ptr<RenderInterface> CreateDeviceThreadInterface(std::shared_ptr<DeviceThread> deviceThread, const std::function<std::unique_ptr<RenderInterface>()> &createInterface) {
        return runSync(deviceThread.get(), [&]() -> std::unique_ptr<RenderInterface> {
            std::unique_ptr<RenderInterface> renderInterface = createInterface();
            return renderInterface ? std::make_unique<DTInterface>(deviceThread, std::move(renderInterface)) : nullptr;
        });
    }

#if defined(__EMSCRIPTEN_PTHREADS__)
    struct EmscriptenDeviceThread : DeviceThread {
        emscripten::ProxyingQueue queue;
        pthread_t thread;

        EmscriptenDeviceThread(pthread_t thread) : thread(thread) { }

        bool isCurrent() const {
            return pthread_equal(pthread_self(), thread);
        }

        void runSync(const std::function<void()> &work) override {
            if (isCurrent()) {
                queue.execute();
                work();
            }
            else {
                [[maybe_unused]] bool proxied = queue.proxySync(thread, work);
                assert(proxied && "The device thread is gone.");
            }
        }

        void runAsync(std::function<void()> &&work) override {
            if (isCurrent()) {
                queue.execute();
                work();
            }
            else {
                [[maybe_unused]] bool proxied = queue.proxyAsync(thread, std::move(work));
                assert(proxied && "The device thread is gone.");
            }
        }
    };

    std::shared_ptr<DeviceThread> CreateEmscriptenDeviceThread(pthread_t thread) {
        return std::make_shared<EmscriptenDeviceThread>(thread);
    }
#endif
};
