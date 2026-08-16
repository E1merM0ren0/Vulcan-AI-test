#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vk {

struct Pipeline {
    VkPipeline handle = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
};

class Context {
public:
    bool init(const std::string& spvDir);
    void shutdown();

    VkBuffer createBuffer(VkDeviceSize size);
    void destroyBuffer(VkBuffer b);
    void uploadBuffer(VkBuffer b, VkDeviceSize size, const void* data);
    void downloadBuffer(VkBuffer b, VkDeviceSize size, void* out);
    void zeroBuffer(VkBuffer b, VkDeviceSize size);

    bool runCompute(const std::string& spvName,
                    uint32_t workX, uint32_t workY, uint32_t workZ,
                    const std::vector<VkBuffer>& bufs,
                    const void* pc, uint32_t pcSize);

    bool loadShader(const std::string& spvName, Pipeline* out);
    const std::string& spvDir() const { return spvDir_; }

private:
    std::vector<char> readFile(const std::string& path);
    VkDeviceMemory allocMemory(VkBuffer b, VkMemoryPropertyFlags flags, VkDeviceSize* sizeOut);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkDescriptorPool dsPool_ = VK_NULL_HANDLE;
    std::string spvDir_;
    std::map<std::string, Pipeline> pipelines_;
};

}
