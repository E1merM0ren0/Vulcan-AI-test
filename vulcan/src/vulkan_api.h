// vulkan_api.h — Minimal Vulkan compute wrapper.
//
// Owns the Vulkan instance/device and a single compute queue. It exposes a
// tiny storage-buffer + push-constant compute API so the training engine can
// dispatch a GLSL shader over a set of already-allocated float buffers without
// touching raw Vulkan objects. Every buffer is host-visible and coherent, so
// upload/download is a plain memcpy through a persistent mapping (no staging,
// no explicit sync) — good enough for the small graphs in this project.
//
// Buffer layout convention: each shader reads/writes zero or more storage
// buffers bound at descriptor binding 0..N (order decided by the caller via
// `bufs`), plus up to 32 bytes of push constants. Workgroup dispatch size is
// chosen per shader by the caller (see graph.cpp's runOp()).

#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vk {

// Identity object for a lazily-compiled SPIR-V compute pipeline.
// All three handles are created together in loadShader() and freed in Context::shutdown().
struct Pipeline {
    VkPipeline handle = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
};

class Context {
public:
    // Create a Vulkan instance + device with a compute-capable queue family and
    // set up the command pool / descriptor pool. `spvDir` is searched for the
    // compiled .spv shaders. Returns false (and prints to stderr) on failure.
    bool init(const std::string& spvDir);
    // Destroy every pipeline, buffer and the device/instance. Idempotent-safe at exit.
    void shutdown();

    // Allocate a host-visible, coherent storage buffer of `size` bytes and keep
    // it persistently mapped until destroyBuffer(). Returns VK_NULL_HANDLE on failure.
    VkBuffer createBuffer(VkDeviceSize size);
    void destroyBuffer(VkBuffer b);
    // memcpy into/out-of the buffer's persistent mapping (no Vulkan sync needed).
    void uploadBuffer(VkBuffer b, VkDeviceSize size, const void* data);
    void downloadBuffer(VkBuffer b, VkDeviceSize size, void* out);
    // memset the persistent mapping to zero.
    void zeroBuffer(VkBuffer b, VkDeviceSize size);

    // Dispatch one compute shader. If `deferred` is true, records into the active
    // command buffer without submitting; caller must call submitAndWait() after
    // all deferred dispatches. If `deferred` is false (default), submits and
    // waits immediately (legacy behavior).
    bool runCompute(const std::string& spvName,
                    uint32_t workX, uint32_t workY, uint32_t workZ,
                    const std::vector<VkBuffer>& bufs,
                    const void* pc, uint32_t pcSize,
                    bool deferred = false,
                    bool barrierAfter = false);
    // Submit the currently recorded command buffer and wait for GPU idle.
    // Only valid after one or more deferred runCompute() calls.
    bool submitAndWait();
    // Begin a new deferred batch (resets command buffer).
    void beginDeferredBatch();

    // Compile spvName.spv into a Pipeline (used internally by runCompute).
    bool loadShader(const std::string& spvName, Pipeline* out);
    const std::string& spvDir() const { return spvDir_; }

private:
    // Read an entire file into memory (used to load .spv shader binaries).
    std::vector<char> readFile(const std::string& path);
    // (unused helper) find a device memory type matching `flags`.
    VkDeviceMemory allocMemory(VkBuffer b, VkMemoryPropertyFlags flags, VkDeviceSize* sizeOut);

    // Handles: one instance, one physical/device, one compute queue + family.
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    // One primary command buffer reused (and reset) for every dispatch.
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    // Descriptor pool shared by all transient descriptor sets.
    VkDescriptorPool dsPool_ = VK_NULL_HANDLE;
    // Directory searched for "<name>.spv" (usually vulcan/build/spv).
    std::string spvDir_;
    // name -> compiled pipeline cache.
    std::map<std::string, Pipeline> pipelines_;
    // Descriptor sets allocated during deferred batch, freed in submitAndWait().
    std::vector<VkDescriptorSet> deferredDescriptorSets_;
};

}
