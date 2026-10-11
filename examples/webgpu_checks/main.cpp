//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

// Checks, in a browser, what the WebGPU backend does where core WebGPU has no equivalent (plume_webgpu.h):
// - a READBACK buffer mapped right after execute and wait holds what the GPU wrote, twice in a row;
// - a formatted R8_UINT buffer of 7 bytes, bound as storage and unpacked by the shader;
// - clears of rectangles of color and depth, with a draw after them that needs its own viewport back;
// - push constants beside 4 descriptor sets, sharing bind group 3 with the 4th;
// - a compute shader writing an rgba8unorm storage texture, which WebGPU only allows write-only;
// - present() paced to animation frames, and the refresh rate it measures.
// Each check prints a line, then "plume WebGPU checks: N of M passed", and the canvas ends green when all passed.
// Its shaders are WGSL written by hand, so it needs no shader tools. Built with -pthread, the checks run on a thread
// of their own through plume_device_thread.h, as RT64 would, and the page's main thread runs the device; presents
// aren't paced then, so that check only counts them.

#include "plume_render_interface.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <emscripten.h>
#ifdef __EMSCRIPTEN_PTHREADS__
#include <thread>
#include "plume_device_thread.h"
#endif

namespace plume {
    extern std::unique_ptr<RenderInterface> CreateWebGPUInterface();
}

using namespace plume;

static const char UnpackShader[] = R"(
@group(0) @binding(0) var<storage, read> bytes: array<u32>;
@group(0) @binding(1) var<storage, read_write> words: array<u32>;
@compute @workgroup_size(8) fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i < arrayLength(&words)) {
        words[i] = (bytes[i / 4u] >> ((i % 4u) * 8u)) & 0xFFu;
    }
}
)";

static const char SharedGroupShader[] = R"(
struct PushConstants { add: u32, slot: u32 }
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(1) @binding(0) var<storage, read> b: array<u32>;
@group(2) @binding(0) var<storage, read> c: array<u32>;
@group(3) @binding(0) var<storage, read_write> sums: array<u32>;
@group(3) @binding(999) var<uniform> pushConstants: PushConstants;
@compute @workgroup_size(1) fn main() {
    sums[pushConstants.slot] = a[0] + b[0] + c[0] + pushConstants.add;
}
)";

static const char StorageTextureShader[] = R"(
@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3u) {
    textureStore(output, id.xy, vec4f(f32(id.x) / 255.0, f32(id.y) / 255.0, 1.0, 1.0));
}
)";

static const char DrawShader[] = R"(
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.5, 1.0);
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 1.0, 1.0);
}
)";

static uint32_t passed = 0;
static uint32_t checked = 0;

static void check(bool ok, const char *name, const char *detail = "") {
    checked++;
    passed += ok ? 1 : 0;
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, (detail[0] != '\0') ? ": " : "", detail);
}

struct Context {
    std::unique_ptr<RenderInterface> renderInterface;
    std::unique_ptr<RenderDevice> device;
    std::unique_ptr<RenderCommandQueue> queue;
    std::unique_ptr<RenderCommandList> commandList;
    std::unique_ptr<RenderCommandFence> fence;

    void execute() {
        commandList->end();
        queue->executeCommandLists(commandList.get(), fence.get());
        queue->waitForCommandFence(fence.get());
    }
};

// Bytes go up in an R8_UINT formatted buffer 7 bytes long, a shader unpacks each into a word, and the words come
// back through a READBACK buffer.
static void checkFormattedBufferAndReadback(Context &ctx) {
    const RenderDescriptorRange ranges[] = {
        RenderDescriptorRange(RenderDescriptorRangeType::FORMATTED_BUFFER, 0, 1),
        RenderDescriptorRange(RenderDescriptorRangeType::READ_WRITE_STRUCTURED_BUFFER, 1, 1)
    };

    const RenderDescriptorSetDesc setDesc(ranges, 2);
    auto layout = ctx.device->createPipelineLayout(RenderPipelineLayoutDesc(nullptr, 0, &setDesc, 1));
    auto shader = ctx.device->createShader(UnpackShader, strlen(UnpackShader), "main", RenderShaderFormat::WGSL);
    auto pipeline = ctx.device->createComputePipeline(RenderComputePipelineDesc(layout.get(), shader.get(), 8, 1, 1));
    auto bytes = ctx.device->createBuffer(RenderBufferDesc::UploadBuffer(7, RenderBufferFlag::FORMATTED));
    auto bytesView = bytes->createBufferFormattedView(RenderFormat::R8_UINT);
    auto words = ctx.device->createBuffer(RenderBufferDesc::DefaultBuffer(8 * sizeof(uint32_t), RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS));
    auto readback = ctx.device->createBuffer(RenderBufferDesc::ReadbackBuffer(8 * sizeof(uint32_t)));
    auto set = ctx.device->createDescriptorSet(setDesc);
    set->setBuffer(0, bytes.get(), 7, nullptr, bytesView.get());
    set->setBuffer(1, words.get(), 8 * sizeof(uint32_t));

    const uint8_t rounds[2][7] = { { 1, 2, 3, 250, 5, 6, 7 }, { 70, 60, 50, 40, 30, 20, 10 } };
    for (uint32_t round = 0; round < 2; round++) {
        memcpy(bytes->map(), rounds[round], 7);
        bytes->unmap();

        ctx.commandList->begin();
        ctx.commandList->setPipeline(pipeline.get());
        ctx.commandList->setComputePipelineLayout(layout.get());
        ctx.commandList->setComputeDescriptorSet(set.get(), 0);
        ctx.commandList->dispatch(1, 1, 1);
        ctx.commandList->copyBufferRegion(readback.get(), words.get(), 8 * sizeof(uint32_t));
        const auto start = std::chrono::steady_clock::now();
        ctx.execute();
        const uint32_t *values = static_cast<const uint32_t *>(readback->map());
        const double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        bool ok = true;
        char detail[160];
        int length = snprintf(detail, sizeof(detail), "%.1f ms, read", waited);
        for (uint32_t i = 0; i < 7; i++) {
            ok = ok && (values[i] == rounds[round][i]);
            length += snprintf(detail + length, sizeof(detail) - length, " %u", values[i]);
        }

        readback->unmap();
        check(ok, (round == 0) ? "R8 formatted buffer, READBACK after execute and wait" : "the same again with new bytes", detail);
    }
}

// Sets 0-2 hold 1, 20 and 300, set 3 the output, and two dispatches add 4000 and 50000 through push constants into
// slots 0 and 1.
static void checkPushConstantsWithFourSets(Context &ctx) {
    const RenderDescriptorRange inputRange(RenderDescriptorRangeType::STRUCTURED_BUFFER, 0, 1);
    const RenderDescriptorRange outputRange(RenderDescriptorRangeType::READ_WRITE_STRUCTURED_BUFFER, 0, 1);
    const RenderDescriptorSetDesc setDescs[] = {
        RenderDescriptorSetDesc(&inputRange, 1), RenderDescriptorSetDesc(&inputRange, 1), RenderDescriptorSetDesc(&inputRange, 1), RenderDescriptorSetDesc(&outputRange, 1)
    };

    const RenderPushConstantRange pushConstantRange(0, 0, 0, 2 * sizeof(uint32_t), RenderShaderStageFlag::COMPUTE);
    auto layout = ctx.device->createPipelineLayout(RenderPipelineLayoutDesc(&pushConstantRange, 1, setDescs, 4));
    auto shader = ctx.device->createShader(SharedGroupShader, strlen(SharedGroupShader), "main", RenderShaderFormat::WGSL);
    auto pipeline = ctx.device->createComputePipeline(RenderComputePipelineDesc(layout.get(), shader.get(), 1, 1, 1));
    const uint32_t values[] = { 1, 20, 300 };
    std::unique_ptr<RenderBuffer> inputs[3];
    std::unique_ptr<RenderDescriptorSet> sets[4];
    for (uint32_t i = 0; i < 3; i++) {
        inputs[i] = ctx.device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(uint32_t), RenderBufferFlag::STORAGE));
        memcpy(inputs[i]->map(), &values[i], sizeof(uint32_t));
        inputs[i]->unmap();
        sets[i] = ctx.device->createDescriptorSet(setDescs[i]);
        sets[i]->setBuffer(0, inputs[i].get(), sizeof(uint32_t));
    }

    auto sums = ctx.device->createBuffer(RenderBufferDesc::DefaultBuffer(2 * sizeof(uint32_t), RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS));
    auto readback = ctx.device->createBuffer(RenderBufferDesc::ReadbackBuffer(2 * sizeof(uint32_t)));
    sets[3] = ctx.device->createDescriptorSet(setDescs[3]);
    sets[3]->setBuffer(0, sums.get(), 2 * sizeof(uint32_t));

    ctx.commandList->begin();
    ctx.commandList->setPipeline(pipeline.get());
    ctx.commandList->setComputePipelineLayout(layout.get());
    for (uint32_t i = 0; i < 4; i++) {
        ctx.commandList->setComputeDescriptorSet(sets[i].get(), i);
    }

    const uint32_t pushConstants[2][2] = { { 4000, 0 }, { 50000, 1 } };
    for (uint32_t i = 0; i < 2; i++) {
        ctx.commandList->setComputePushConstants(0, pushConstants[i], 0, sizeof(pushConstants[i]));
        ctx.commandList->dispatch(1, 1, 1);
    }

    ctx.commandList->copyBufferRegion(readback.get(), sums.get(), 2 * sizeof(uint32_t));
    ctx.execute();
    const uint32_t *results = static_cast<const uint32_t *>(readback->map());
    char detail[96];
    snprintf(detail, sizeof(detail), "read %u %u", results[0], results[1]);
    check((results[0] == 4321) && (results[1] == 50321), "push constants sharing group 3 with a 4th descriptor set", detail);
    readback->unmap();
}

// A compute shader writes each texel of a 64x64 rgba8unorm storage texture with its own coordinates, and the
// texture comes back through a READBACK buffer.
static void checkStorageTexture(Context &ctx) {
    const uint32_t Size = 64;
    const RenderDescriptorRange ranges[] = { RenderDescriptorRange(RenderDescriptorRangeType::READ_WRITE_TEXTURE, 0, 1) };
    const RenderDescriptorSetDesc setDesc(ranges, 1);
    auto layout = ctx.device->createPipelineLayout(RenderPipelineLayoutDesc(nullptr, 0, &setDesc, 1));
    auto shader = ctx.device->createShader(StorageTextureShader, strlen(StorageTextureShader), "main", RenderShaderFormat::WGSL);
    auto pipeline = ctx.device->createComputePipeline(RenderComputePipelineDesc(layout.get(), shader.get(), 8, 8, 1));
    auto texture = ctx.device->createTexture(RenderTextureDesc::Texture2D(Size, Size, 1, RenderFormat::R8G8B8A8_UNORM, RenderTextureFlag::STORAGE | RenderTextureFlag::UNORDERED_ACCESS));
    auto readback = ctx.device->createBuffer(RenderBufferDesc::ReadbackBuffer(Size * Size * 4));
    auto set = ctx.device->createDescriptorSet(setDesc);
    set->setTexture(0, texture.get(), RenderTextureLayout::GENERAL);

    ctx.commandList->begin();
    ctx.commandList->setPipeline(pipeline.get());
    ctx.commandList->setComputePipelineLayout(layout.get());
    ctx.commandList->setComputeDescriptorSet(set.get(), 0);
    ctx.commandList->dispatch(Size / 8, Size / 8, 1);
    ctx.commandList->copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(readback.get(), RenderFormat::R8G8B8A8_UNORM, Size, Size, 1, Size), RenderTextureCopyLocation::Subresource(texture.get()));
    ctx.execute();

    const uint32_t *texels = static_cast<const uint32_t *>(readback->map());
    uint32_t wrong = 0;
    for (uint32_t y = 0; y < Size; y++) {
        for (uint32_t x = 0; x < Size; x++) {
            wrong += (texels[y * Size + x] != (0xFFFF0000U | (y << 8) | x)) ? 1 : 0;
        }
    }

    char detail[160];
    snprintf(detail, sizeof(detail), "(5,9) %08X (63,63) %08X, %u of %u texels wrong", texels[9 * Size + 5], texels[Size * Size - 1], wrong, Size * Size);
    check(wrong == 0, "an rgba8unorm storage texture written by a compute shader", detail);
    readback->unmap();
}

// Full clears to red and depth 1, then green rectangles, depth 0.25 over the bottom half (asking for the stencil too,
// as clearDepthStencil's defaults do, which D32_FLOAT hasn't got), and a blue triangle at
// depth 0.5 tested LESS over everything: blue on top, red and green kept below, and the depth it wrote is 0.5, which
// it only is if the clears gave the draw its viewport back.
static void checkRectangleClears(Context &ctx) {
    const uint32_t Size = 64;
    auto color = ctx.device->createTexture(RenderTextureDesc::ColorTarget(Size, Size, RenderFormat::R8G8B8A8_UNORM));
    auto depth = ctx.device->createTexture(RenderTextureDesc::DepthTarget(Size, Size, RenderFormat::D32_FLOAT));
    const RenderTexture *colorTargets[] = { color.get() };
    auto framebuffer = ctx.device->createFramebuffer(RenderFramebufferDesc(colorTargets, 1, depth.get()));
    auto layout = ctx.device->createPipelineLayout(RenderPipelineLayoutDesc());
    auto vertexShader = ctx.device->createShader(DrawShader, strlen(DrawShader), "vs", RenderShaderFormat::WGSL);
    auto pixelShader = ctx.device->createShader(DrawShader, strlen(DrawShader), "fs", RenderShaderFormat::WGSL);
    RenderGraphicsPipelineDesc pipelineDesc;
    pipelineDesc.pipelineLayout = layout.get();
    pipelineDesc.vertexShader = vertexShader.get();
    pipelineDesc.pixelShader = pixelShader.get();
    pipelineDesc.depthEnabled = true;
    pipelineDesc.depthWriteEnabled = true;
    pipelineDesc.depthFunction = RenderComparisonFunction::LESS;
    pipelineDesc.depthTargetFormat = RenderFormat::D32_FLOAT;
    pipelineDesc.renderTargetFormat[0] = RenderFormat::R8G8B8A8_UNORM;
    pipelineDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
    pipelineDesc.renderTargetCount = 1;
    auto pipeline = ctx.device->createGraphicsPipeline(pipelineDesc);
    auto colorReadback = ctx.device->createBuffer(RenderBufferDesc::ReadbackBuffer(Size * Size * 4));
    auto depthReadback = ctx.device->createBuffer(RenderBufferDesc::ReadbackBuffer(Size * Size * 4));

    const RenderRect greenRects[] = { RenderRect(8, 8, 24, 24), RenderRect(40, 40, 56, 56) };
    const RenderRect bottomHalf(0, Size / 2, Size, Size);
    ctx.commandList->begin();
    ctx.commandList->setFramebuffer(framebuffer.get());
    ctx.commandList->setGraphicsPipelineLayout(layout.get());
    ctx.commandList->setPipeline(pipeline.get());
    ctx.commandList->setViewports(RenderViewport(0.0f, 0.0f, float(Size), float(Size)));
    ctx.commandList->setScissors(RenderRect(0, 0, Size, Size));
    ctx.commandList->clearColor(0, RenderColor(1.0f, 0.0f, 0.0f, 1.0f));
    ctx.commandList->clearDepth(true, 1.0f);
    ctx.commandList->clearColor(0, RenderColor(0.0f, 1.0f, 0.0f, 1.0f), greenRects, 2);
    ctx.commandList->clearDepthStencil(true, true, 0.25f, 0, &bottomHalf, 1);
    ctx.commandList->drawInstanced(3, 1, 0, 0);
    ctx.commandList->setFramebuffer(nullptr);
    ctx.commandList->copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(colorReadback.get(), RenderFormat::R8G8B8A8_UNORM, Size, Size, 1, Size), RenderTextureCopyLocation::Subresource(color.get()));
    ctx.commandList->copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(depthReadback.get(), RenderFormat::D32_FLOAT, Size, Size, 1, Size), RenderTextureCopyLocation::Subresource(depth.get()));
    ctx.execute();

    const uint32_t *pixels = static_cast<const uint32_t *>(colorReadback->map());
    const float *depths = static_cast<const float *>(depthReadback->map());
    auto pixel = [&](uint32_t x, uint32_t y) { return pixels[y * Size + x]; };
    auto depthAt = [&](uint32_t x, uint32_t y) { return depths[y * Size + x]; };
    const uint32_t Red = 0xFF0000FF, Green = 0xFF00FF00, Blue = 0xFFFF0000;
    char detail[160];
    snprintf(detail, sizeof(detail), "(16,16) %08X (48,48) %08X (4,60) %08X (60,4) %08X", pixel(16, 16), pixel(48, 48), pixel(4, 60), pixel(60, 4));
    check((pixel(48, 48) == Green) && (pixel(4, 60) == Red) && (pixel(39, 39) == Red) && (pixel(56, 56) == Red), "color rectangles cleared, the rest kept", detail);
    check((pixel(16, 16) == Blue) && (pixel(60, 4) == Blue) && (pixel(48, 48) == Green), "a draw after them, blocked where the depth rectangle was cleared", detail);
    snprintf(detail, sizeof(detail), "top %.3f, bottom %.3f", depthAt(10, 10), depthAt(10, 50));
    check((std::fabs(depthAt(10, 10) - 0.5f) < 1e-4f) && (std::fabs(depthAt(10, 50) - 0.25f) < 1e-4f), "depth: the draw's 0.5 above the cleared 0.25", detail);
    colorReadback->unmap();
    depthReadback->unmap();
}

// Kept, since unconfiguring the canvas when it goes would clear it.
static std::unique_ptr<RenderSwapChain> swapChain;

// Presents 30 frames, each cleared to the summary color, and times them.
static void checkPresent(Context &ctx) {
    swapChain = ctx.queue->createSwapChain(RenderSwapChainDesc("#canvas", RenderFormat::B8G8R8A8_UNORM, 2));
    swapChain->resize();
    const RenderColor summary = (passed == checked) ? RenderColor(0.0f, 0.6f, 0.2f) : RenderColor(0.8f, 0.0f, 0.0f);
    const uint32_t Frames = 30;
    uint32_t presented = 0;
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < Frames; i++) {
        uint32_t textureIndex = 0;
        if (!swapChain->acquireTexture(nullptr, &textureIndex)) {
            continue;
        }

        const RenderTexture *target = swapChain->getTexture(textureIndex);
        auto framebuffer = ctx.device->createFramebuffer(RenderFramebufferDesc(&target, 1));
        ctx.commandList->begin();
        ctx.commandList->setFramebuffer(framebuffer.get());
        ctx.commandList->clearColor(0, summary);
        ctx.commandList->setFramebuffer(nullptr);
        ctx.execute();
        presented += swapChain->present(textureIndex, nullptr, 0) ? 1 : 0;
    }

    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const uint32_t refreshRate = swapChain->getRefreshRate();
    char detail[160];
    snprintf(detail, sizeof(detail), "%u frames in %.0f ms, refresh rate %u Hz", presented, elapsed, refreshRate);
#ifdef __EMSCRIPTEN_PTHREADS__
    check(presented == Frames, "presents through the device thread", detail);
#else
    const double expected = 1000.0 * Frames / refreshRate;
    check((presented == Frames) && (elapsed > 0.8 * expected) && (refreshRate >= 24), "present paced to animation frames", detail);
#endif
}

// Kept, like the swap chain.
static Context ctx;

static void runChecks(std::unique_ptr<RenderInterface> renderInterface) {
    ctx.renderInterface = std::move(renderInterface);
    ctx.device = ctx.renderInterface->createDevice();
    if (ctx.device == nullptr) {
        printf("FAIL no WebGPU device\n");
        return;
    }

    ctx.queue = ctx.device->createCommandQueue(RenderCommandListType::DIRECT);
    ctx.commandList = ctx.queue->createCommandList();
    ctx.fence = ctx.device->createCommandFence();
#ifdef __EMSCRIPTEN_PTHREADS__
    printf("this build runs the checks on a thread of their own, through the device thread\n");
#else
    printf("this build %s wait (emscripten_has_asyncify() = %d)\n", emscripten_has_asyncify() ? "can" : "can't", emscripten_has_asyncify());
#endif
    checkFormattedBufferAndReadback(ctx);
    checkPushConstantsWithFourSets(ctx);
    checkStorageTexture(ctx);
    checkRectangleClears(ctx);
    checkPresent(ctx);
    printf("plume WebGPU checks: %u of %u passed\n", passed, checked);
}

int main(int argc, char *argv[]) {
#ifdef __EMSCRIPTEN_PTHREADS__
    // This is the page's main thread, which owns the device: it goes back to its event loop and runs the device's work.
    auto renderInterface = CreateDeviceThreadInterface(CreateEmscriptenDeviceThread(pthread_self()), []() { return CreateWebGPUInterface(); });
    std::thread([renderInterface = std::move(renderInterface)]() mutable { runChecks(std::move(renderInterface)); }).detach();
#else
    runChecks(CreateWebGPUInterface());
#endif
    return 0;
}
