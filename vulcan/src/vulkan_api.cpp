#include "vulkan_api.h"
#include <cstring>
#include <fstream>
#include <iostream>

namespace vk {

struct BufferAlloc {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    void* mapped = nullptr;
};

static std::vector<BufferAlloc> g_alloc;

bool Context::init(const std::string& spvDir) {
    spvDir_ = spvDir;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instInfo, nullptr, &instance_) != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed\n";
        return false;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "no Vulkan devices\n";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());
    phys_ = devices[0];

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &familyCount, families.data());

    bool found = false;
    for (uint32_t i = 0; i < familyCount; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamily_ = i;
            found = true;
            break;
        }
    }
    if (!found) {
        std::cerr << "no compute queue family\n";
        return false;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qInfo{};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = queueFamily_;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures features{};
    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.pQueueCreateInfos = &qInfo;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pEnabledFeatures = &features;

    if (vkCreateDevice(phys_, &devInfo, nullptr, &device_) != VK_SUCCESS) {
        std::cerr << "vkCreateDevice failed\n";
        return false;
    }
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily_;
    vkCreateCommandPool(device_, &poolInfo, nullptr, &pool_);

    VkCommandBufferAllocateInfo cmdInfo{};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdInfo.commandPool = pool_;
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &cmdInfo, &cmd_);

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 128;

    VkDescriptorPoolCreateInfo dpInfo{};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpInfo.maxSets = 64;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(device_, &dpInfo, nullptr, &dsPool_);

    return true;
}

void Context::shutdown() {
    for (auto& a : g_alloc) {
        if (a.mapped) vkUnmapMemory(device_, a.memory);
        vkDestroyBuffer(device_, a.buffer, nullptr);
        vkFreeMemory(device_, a.memory, nullptr);
    }
    g_alloc.clear();
    for (auto& kv : pipelines_) {
        vkDestroyPipeline(device_, kv.second.handle, nullptr);
        vkDestroyPipelineLayout(device_, kv.second.layout, nullptr);
        vkDestroyDescriptorSetLayout(device_, kv.second.dsLayout, nullptr);
    }
    pipelines_.clear();
    if (dsPool_) vkDestroyDescriptorPool(device_, dsPool_, nullptr);
    if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
    if (device_) vkDestroyDevice(device_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    dsPool_ = VK_NULL_HANDLE;
}

VkBuffer Context::createBuffer(VkDeviceSize size) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    vkCreateBuffer(device_, &info, nullptr, &buffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer, &req);

    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phys_, &props);

    uint32_t memType = 0;
    bool found = false;
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memType = i;
            found = true;
            break;
        }
    }
    if (!found) {
        std::cerr << "no host-visible memory type\n";
        vkDestroyBuffer(device_, buffer, nullptr);
        return VK_NULL_HANDLE;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = req.size;
    allocInfo.memoryTypeIndex = memType;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "vkAllocateMemory failed\n";
        vkDestroyBuffer(device_, buffer, nullptr);
        return VK_NULL_HANDLE;
    }
    vkBindBufferMemory(device_, buffer, memory, 0);

    void* mapped = nullptr;
    vkMapMemory(device_, memory, 0, req.size, 0, &mapped);

    BufferAlloc alloc{};
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.size = size;
    alloc.mapped = mapped;
    g_alloc.push_back(alloc);
    return buffer;
}

void Context::destroyBuffer(VkBuffer b) {
    for (size_t i = 0; i < g_alloc.size(); i++) {
        if (g_alloc[i].buffer == b) {
            vkUnmapMemory(device_, g_alloc[i].memory);
            vkDestroyBuffer(device_, b, nullptr);
            vkFreeMemory(device_, g_alloc[i].memory, nullptr);
            g_alloc.erase(g_alloc.begin() + (long)i);
            return;
        }
    }
}

void Context::uploadBuffer(VkBuffer b, VkDeviceSize size, const void* data) {
    for (auto& a : g_alloc) {
        if (a.buffer == b) {
            std::memcpy(a.mapped, data, (size_t)size);
            return;
        }
    }
}

void Context::downloadBuffer(VkBuffer b, VkDeviceSize size, void* out) {
    for (auto& a : g_alloc) {
        if (a.buffer == b) {
            std::memcpy(out, a.mapped, (size_t)size);
            return;
        }
    }
}

void Context::zeroBuffer(VkBuffer b, VkDeviceSize size) {
    for (auto& a : g_alloc) {
        if (a.buffer == b) {
            std::memset(a.mapped, 0, (size_t)size);
            return;
        }
    }
}

std::vector<char> Context::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> data;
    if (!file.is_open()) return data;
    file.seekg(0, std::ios::end);
    std::streamoff len = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize((size_t)len);
    file.read(data.data(), len);
    return data;
}

bool Context::loadShader(const std::string& spvName, Pipeline* out) {
    std::vector<std::string> dirs;
    dirs.push_back(spvDir_);
    dirs.push_back(spvDir_ + "/");
    for (const auto& d : dirs) {
        std::string path = d + spvName + ".spv";
        std::vector<char> code = readFile(path);
        if (code.empty()) continue;

        VkShaderModuleCreateInfo modInfo{};
        modInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        modInfo.codeSize = code.size();
        modInfo.pCode = (const uint32_t*)code.data();

        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &modInfo, nullptr, &module) != VK_SUCCESS) {
            std::cerr << "vkCreateShaderModule failed for " << spvName << "\n";
            return false;
        }

        const int MAX_BINDINGS = 8;
        std::vector<VkDescriptorSetLayoutBinding> bindings(MAX_BINDINGS);
        for (int i = 0; i < MAX_BINDINGS; i++) {
            bindings[i].binding = (uint32_t)i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dslInfo{};
        dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslInfo.bindingCount = (uint32_t)bindings.size();
        dslInfo.pBindings = bindings.data();

        VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(device_, &dslInfo, nullptr, &dsLayout);

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = 32;

        VkPipelineLayoutCreateInfo plInfo{};
        plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &dsLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pcRange;

        VkPipelineLayout layout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(device_, &plInfo, nullptr, &layout);

        VkComputePipelineCreateInfo cpInfo{};
        cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = module;
        cpInfo.stage.pName = "main";
        cpInfo.layout = layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipeline);

        vkDestroyShaderModule(device_, module, nullptr);

        out->handle = pipeline;
        out->layout = layout;
        out->dsLayout = dsLayout;
        return true;
    }
    std::cerr << "shader not found: " << spvName << "\n";
    return false;
}

bool Context::runCompute(const std::string& spvName,
                         uint32_t workX, uint32_t workY, uint32_t workZ,
                         const std::vector<VkBuffer>& bufs,
                         const void* pc, uint32_t pcSize) {
    Pipeline pipeline;
    auto it = pipelines_.find(spvName);
    if (it == pipelines_.end()) {
        if (!loadShader(spvName, &pipeline)) return false;
        pipelines_[spvName] = pipeline;
    } else {
        pipeline = it->second;
    }

    VkDescriptorSetAllocateInfo dsInfo{};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = dsPool_;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &pipeline.dsLayout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device_, &dsInfo, &set) != VK_SUCCESS) {
        std::cerr << "vkAllocateDescriptorSets failed\n";
        return false;
    }

    std::vector<VkDescriptorBufferInfo> bufferInfos(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); i++) {
        bufferInfos[i].buffer = bufs[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device_, (uint32_t)writes.size(), writes.data(), 0, nullptr);

    vkResetCommandBuffer(cmd_, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_, &begin);

    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
    if (pcSize > 0) {
        vkCmdPushConstants(cmd_, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
    }
    vkCmdDispatch(cmd_, workX, workY, workZ);

    vkEndCommandBuffer(cmd_);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_;

    vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);
    vkFreeDescriptorSets(device_, dsPool_, 1, &set);

    return true;
}

}
