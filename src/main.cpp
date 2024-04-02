#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 
#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <optional>
#include <set>
#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp
#include <fstream>
#include <filesystem>
#include <array>
#define STB_IMAGE_IMPLEMENTATION
#include <src/stb_image.h>

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    std::cout<<std::filesystem::current_path().string()<<std::endl;

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}


struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{   .binding    = 0,
                                                                .stride     = sizeof(Vertex),
                                                                .inputRate  = vk::VertexInputRate::eVertex};
        
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};
const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(4) float time;
};

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const std::vector<const char*> validationLayers = {
"VK_LAYER_KHRONOS_validation","VK_LAYER_LUNARG_monitor"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_MAINTENANCE1_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif
const int MAX_FRAMES_IN_FLIGHT = 2;
class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;


    vk::Instance instance;
    vk::Device device;
    vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    VkDebugUtilsMessengerEXT debugMessenger;
    vk::SurfaceKHR surface;

    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;

    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    std::vector<vk::Framebuffer> swapChainFramebuffers;
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence>  inFlightFences;
    uint32_t currentFrame = 0;
    bool framebufferResized = false;

    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    vk::Image textureImage;
    vk::DeviceMemory textureImageMemory;
    vk::ImageView textureImageView;
    vk::Sampler textureSampler;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void drawFrame(){
        device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        auto acquireResult= device.acquireNextImageKHR(
            swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE
        );
        if (acquireResult.result == vk::Result::eSuccess 
            ||  acquireResult.result == vk::Result::eSuboptimalKHR) {
            imageIndex = acquireResult.value;
        }
        else if (acquireResult.result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        else{
            throw std::runtime_error("failed to acquire next image");
        }
        device.resetFences(inFlightFences[currentFrame]);
        commandBuffers[currentFrame].reset();
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        updateUniformBuffer(currentFrame);
        vk::SubmitInfo submitInfo{  .waitSemaphoreCount     =   1,
                                    .pWaitSemaphores        =   waitSemaphores,
                                    .pWaitDstStageMask      =   waitStages, 
                                    .commandBufferCount     =   1,
                                    .pCommandBuffers        =   &commandBuffers[currentFrame],
                                    .signalSemaphoreCount   =   1,
                                    .pSignalSemaphores      =   signalSemaphores };
        graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);
        vk::SwapchainKHR swapChains[] = { swapChain };
        vk::PresentInfoKHR presentInfo{ .waitSemaphoreCount =   1,
                                        .pWaitSemaphores    =   signalSemaphores,
                                        .swapchainCount     =   1,
                                        .pSwapchains        =   swapChains,
                                        .pImageIndices      =   &imageIndex,
                                        .pResults           =   nullptr };
        auto presentResult = presentQueue.presentKHR(presentInfo);
        if ( (presentResult == vk::Result::eSuboptimalKHR) || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        //currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        currentFrame = (currentFrame + 1)& (MAX_FRAMES_IN_FLIGHT - 1);
    }

    void initVulkan() { 
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createTextureBuffer();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createCommandBuffer();
        createDescriptorPool();
        createDescriptorSets();
        createSyncObjects();
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

        vk::SamplerCreateInfo samplerInfo{  .magFilter              = vk::Filter::eLinear,
                                            .minFilter              = vk::Filter::eLinear,
                                            .mipmapMode             = vk::SamplerMipmapMode::eLinear,
                                            .addressModeU           = vk::SamplerAddressMode::eRepeat,
                                            .addressModeV           = vk::SamplerAddressMode::eRepeat,
                                            .addressModeW           = vk::SamplerAddressMode::eRepeat,
                                            .mipLodBias             = 0.0f,
                                            .anisotropyEnable       = vk::True,
                                            .maxAnisotropy          = properties.limits.maxSamplerAnisotropy,
                                            .compareEnable          = vk::False,
                                            .compareOp              = vk::CompareOp::eAlways,
                                            .minLod                 = 0.0f,
                                            .maxLod                 = 0.0f,
                                            .borderColor            = vk::BorderColor::eIntOpaqueBlack,
                                            .unnormalizedCoordinates= vk::False};
        textureSampler = device.createSampler(samplerInfo);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
        
    }

    vk::ImageView createImageView(vk::Image image, vk::Format format) {
        vk::ImageSubresourceRange subresourceRange{ .aspectMask = vk::ImageAspectFlagBits::eColor,
                                                    .baseMipLevel = 0,
                                                    .levelCount = 1,
                                                    .baseArrayLayer = 0,
                                                    .layerCount = 1 };
        vk::ImageViewCreateInfo viewInfo{   .image = image,
                                            .viewType = vk::ImageViewType::e2D,
                                            .format = format,
                                            .subresourceRange=subresourceRange};
        vk::ImageView imageView = device.createImageView(viewInfo);
        return imageView;
    }

    void createTextureBuffer() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load("../textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;   
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer, stagingBufferMemory);

        void* data = device.mapMemory(stagingBufferMemory, 0, imageSize);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        device.unmapMemory(stagingBufferMemory);
        stbi_image_free(pixels);

        createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

        transitionImageLayout(textureImage,vk::Format::eR8G8B8A8Srgb,vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(stagingBuffer, textureImage,
            static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);

    }
    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::ImageSubresourceLayers subresource{ .aspectMask     = vk::ImageAspectFlagBits::eColor ,
                                                .mipLevel       = 0,
                                                .baseArrayLayer = 0,
                                                .layerCount     = 1 };
        vk::BufferImageCopy region{ .bufferOffset       = 0,
                                    .bufferRowLength    = 0,
                                    .bufferImageHeight  = 0,
                                    .imageSubresource   = subresource,
                                    .imageOffset        = { 0, 0, 0 },
                                    .imageExtent        = { width,height,1} };
        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    void transitionImageLayout(vk::Image image, vk::Format format,
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {

        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
        vk::ImageSubresourceRange subresourceRange{ .aspectMask     = vk::ImageAspectFlagBits::eColor,
                                                    .baseMipLevel   = 0,
                                                    .levelCount     = 1,
                                                    .baseArrayLayer = 0,
                                                    .layerCount     = 1, };
        vk::ImageMemoryBarrier barrier{ .srcAccessMask = {}, // TODO
                                        .dstAccessMask = {}, // TODO
                                        .oldLayout          = oldLayout,
                                        .newLayout          = newLayout,
                                        .srcQueueFamilyIndex= vk::QueueFamilyIgnored,
                                        .dstQueueFamilyIndex= vk::QueueFamilyIgnored,
                                        .image              = image,
                                        .subresourceRange   = subresourceRange};
        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;
        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage =vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(commandBuffer);

    }

    vk::CommandBuffer beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo{.commandPool        = commandPool,
                                                .level              = vk::CommandBufferLevel::ePrimary,
                                                .commandBufferCount = 1 };

        vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

        vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{  .commandBufferCount = 1,
                                    .pCommandBuffers = &commandBuffer };
        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();
        device.freeCommandBuffers(commandPool,commandBuffer);
    }


    void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
        vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image,
        vk::DeviceMemory& imageMemory) {
        vk::ImageCreateInfo imageInfo{  .imageType      = vk::ImageType::e2D,
                                        .format         = format,
                                        .mipLevels      = 1,
                                        .arrayLayers    = 1,
                                        .samples        = vk::SampleCountFlagBits::e1,
                                        .tiling         = tiling,
                                        .usage          = usage,
                                        .sharingMode    = vk::SharingMode::eExclusive,
                                        .initialLayout  = vk::ImageLayout::eUndefined };
        imageInfo.extent.width = static_cast<uint32_t>(width);
        imageInfo.extent.height = static_cast<uint32_t>(height);
        imageInfo.extent.depth = 1;


        image = device.createImage(imageInfo);

        vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);
        vk::MemoryAllocateInfo allocInfo{ .allocationSize = memRequirements.size,
                                            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
                                            properties) };
        imageMemory = device.allocateMemory(allocInfo);

        device.bindImageMemory(image, imageMemory,0);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{ .descriptorPool = descriptorPool,
                                                .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
                                                .pSetLayouts = layouts.data() };
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        descriptorSets = device.allocateDescriptorSets(allocInfo);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DescriptorBufferInfo bufferInfo{.buffer = uniformBuffers[i],
                                                .offset = 0,    
                                                .range  = sizeof(UniformBufferObject) };
            vk::WriteDescriptorSet descriptorWrite{ .dstSet             = descriptorSets[i],
                                                    .dstBinding         = 0,
                                                    .dstArrayElement    = 0,
                                                    .descriptorCount    = 1,
                                                    .descriptorType     = vk::DescriptorType::eUniformBuffer,
                                                    .pImageInfo         = nullptr, // Optional
                                                    .pBufferInfo        = &bufferInfo,
                                                    .pTexelBufferView   = nullptr }; // Optional
            device.updateDescriptorSets(descriptorWrite,0);
        }
    }

    void createDescriptorPool() {
        vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eUniformBuffer,
                                        .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) };
        vk::DescriptorPoolCreateInfo poolInfo{  .maxSets        = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
                                                .poolSizeCount  = 1,
                                                .pPoolSizes     = &poolSize};
        descriptorPool = device.createDescriptorPool(poolInfo);
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        UniformBufferObject ubo{ .model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                                .view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                                .proj = glm::perspective(glm::radians(45.0f),
                                    swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f),
                                .time = time
        };
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent,
                uniformBuffers[i],
                uniformBuffersMemory[i]);

            uniformBuffersMapped[i] = device.mapMemory(uniformBuffersMemory[i], 0, bufferSize);
            
        }
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding{.binding            = 0,
                                                        .descriptorType     = vk::DescriptorType::eUniformBuffer,
                                                        .descriptorCount    = 1,
                                                        .stageFlags         = vk::ShaderStageFlagBits::eVertex,
                                                        .pImmutableSamplers = nullptr };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{   .bindingCount   = 1,
                                                        .pBindings      = &uboLayoutBinding };

        descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);

    }

    void createIndexBuffer(){
        vk::DeviceSize bufferSize= sizeof(indices[0]) * indices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;



        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer, stagingBufferMemory);


        void* data;
        data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, indices.data(),(size_t)bufferSize);
        device.unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize,vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,indexBuffer,indexBufferMemory);

        copyBuffer(stagingBuffer,indexBuffer,bufferSize);
        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }


    void createVertexBuffer() {
        vk::DeviceSize bufferSize= sizeof(vertices[0]) * vertices.size();
        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;



        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuffer, stagingBufferMemory);


        void* data;
        data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, vertices.data(),(size_t)bufferSize);
        device.unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize,vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,vertexBuffer,vertexBufferMemory);

        copyBuffer(stagingBuffer,vertexBuffer,bufferSize);
        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);

    }

    void createBuffer(vk::DeviceSize size, 
     vk::BufferUsageFlags usage, 
     vk::MemoryPropertyFlags properties,
     vk::Buffer& buffer,
     vk::DeviceMemory& bufferMemory)
     {
        vk::BufferCreateInfo bufferInfo{.size = size,
                                        .usage = usage,   
                                        .sharingMode =vk::SharingMode::eExclusive };
        buffer = device.createBuffer(bufferInfo);

        vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
       

        vk::MemoryAllocateInfo allocInfo{   .allocationSize = memRequirements.size,
                                            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,properties)};

        bufferMemory = device.allocateMemory(allocInfo);

        device.bindBufferMemory(buffer, bufferMemory,0);
    }



    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferCopy copyRegion{.size = size };
        commandBuffer.copyBuffer(srcBuffer, dstBuffer,copyRegion);
            
        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    void cleanupSwapChain() {
        for (auto& framebuffer : swapChainFramebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        for (auto& imageViews : swapChainImageViews) {
            device.destroyImageView(imageViews);
        }
        device.destroySwapchainKHR(swapChain);
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            if (glfwWindowShouldClose(window))
                return;
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void cleanup() {
        cleanupSwapChain();
        device.destroySampler(textureSampler);
        device.destroyImageView(textureImageView);
        device.destroyImage(textureImage);
        device.freeMemory(textureImageMemory);
        device.destroyBuffer(vertexBuffer);
        device.freeMemory(vertexBufferMemory);
        device.destroyBuffer(indexBuffer);
        device.freeMemory(indexBufferMemory);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            device.destroySemaphore(imageAvailableSemaphores[i]);
            device.destroySemaphore(renderFinishedSemaphores[i]);
            device.destroyFence(inFlightFences[i]);
        }
        device.destroyCommandPool(commandPool);
        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device.destroyBuffer(uniformBuffers[i]);
            device.freeMemory(uniformBuffersMemory[i]);
        }

        device.destroyDescriptorPool(descriptorPool);
        device.destroyDescriptorSetLayout(descriptorSetLayout);
        device.destroy();
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        instance.destroySurfaceKHR(surface);


        instance.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
    }


    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        vk::SemaphoreCreateInfo semaphoreInfo{};

        vk::FenceCreateInfo fenceInfo{ .flags = vk::FenceCreateFlagBits::eSignaled};
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {

            imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
            renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
            inFlightFences[i] = device.createFence(fenceInfo);
        }
    }

    void recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex) {
        vk::ClearValue clearColor{ std::array<float,4>{0.f,0.f,0.f,0.f} };

        vk::CommandBufferBeginInfo beginInfo{.pInheritanceInfo = nullptr };
        commandBuffer.begin(beginInfo);
        vk::RenderPassBeginInfo renderPassInfo{ .renderPass     =   renderPass,
                                                .framebuffer    =   swapChainFramebuffers[imageIndex],
                                                .clearValueCount=   1,
                                                .pClearValues   =   &clearColor};
        renderPassInfo.renderArea.offset = vk::Offset2D( 0, 0 );
        renderPassInfo.renderArea.extent = swapChainExtent;
        commandBuffer.beginRenderPass(renderPassInfo,vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
        vk::Buffer vertexBuffers[] = { vertexBuffer };
        vk::DeviceSize offsets[] = { 0 };

        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.bindIndexBuffer(indexBuffer,0,vk::IndexType::eUint16);

        vk::Viewport viewport{  .x          =   0.0f,
                                .y          =   static_cast<float>(swapChainExtent.height),
                                .width      =   static_cast<float>(swapChainExtent.width),
                                .height     =   -static_cast<float>(swapChainExtent.height),
                                .minDepth   =   0.0f,
                                .maxDepth   =   1.0f };
        commandBuffer.setViewport(0,viewport);

        vk::Rect2D scissor{ .offset =   { 0, 0 },
                            .extent =   swapChainExtent };
        commandBuffer.setScissor(0, scissor);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,0,descriptorSets[currentFrame], nullptr);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()),1,0,0,0);
        commandBuffer.endRenderPass();
        commandBuffer.end();
    }

    void createCommandBuffer() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool,
                                                .level = vk::CommandBufferLevel::ePrimary,
                                                .commandBufferCount = (uint32_t)commandBuffers.size() };
        commandBuffers = device.allocateCommandBuffers(allocInfo);
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        vk::CommandPoolCreateInfo poolInfo{ .flags              =   vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                            .queueFamilyIndex   =   queueFamilyIndices.graphicsFamily.value() };
        commandPool = device.createCommandPool(poolInfo);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            vk::ImageView attachments[] = {
                swapChainImageViews[i]
            };

            vk::FramebufferCreateInfo framebufferInfo{  .renderPass     = renderPass,
                                                        .attachmentCount= 1,
                                                        .pAttachments   = attachments,
                                                        .width          = swapChainExtent.width,
                                                        .height         = swapChainExtent.height,
                                                        .layers         = 1 };
            
            swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
        }
    }
    void createRenderPass() {
        vk::AttachmentDescription colorAttachment{  .format         =   swapChainImageFormat,
                                                    .samples        =   vk::SampleCountFlagBits::e1,
                                                    .loadOp         =   vk::AttachmentLoadOp::eClear,
                                                    .storeOp        =   vk::AttachmentStoreOp::eStore,
                                                    .stencilLoadOp  =   vk::AttachmentLoadOp::eDontCare,
                                                    .stencilStoreOp =   vk::AttachmentStoreOp::eDontCare,
                                                    .initialLayout  =   vk::ImageLayout::eUndefined,
                                                    .finalLayout    =   vk::ImageLayout::ePresentSrcKHR };

        vk::AttachmentReference colorAttachmentRef{ .attachment =   0,
                                                    .layout     =   vk::ImageLayout::eColorAttachmentOptimal};

        vk::SubpassDescription subpass{ .pipelineBindPoint      =   vk::PipelineBindPoint::eGraphics,
                                        .colorAttachmentCount   =   1,
                                        .pColorAttachments      =   &colorAttachmentRef };

        vk::SubpassDependency dependency{   .srcSubpass     =   vk::SubpassExternal,
                                            .dstSubpass     =   0,
                                            .srcStageMask   =   vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                            .dstStageMask   =   vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                            .dstAccessMask  =   vk::AccessFlagBits::eColorAttachmentWrite };

        vk::RenderPassCreateInfo renderPassInfo{.attachmentCount=   1,
                                                .pAttachments   =   &colorAttachment,
                                                .subpassCount   =   1,
                                                .pSubpasses     =   &subpass, 
                                                .dependencyCount=   1,
                                                .pDependencies  =   &dependency };
        renderPass = device.createRenderPass(renderPassInfo);
    }

    void createGraphicsPipeline() {
        auto shaders_path = std::filesystem::current_path();
        auto vertShaderCode = readFile("../shaders/shader.vert.spv");
        auto fragShaderCode = readFile("../shaders/shader.frag.spv");
        vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .sType   = vk::StructureType::ePipelineShaderStageCreateInfo,
                                                                .stage  = vk::ShaderStageFlagBits::eVertex,
                                                                .module = vertShaderModule,
                                                                .pName  = "main" };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .sType   = vk::StructureType::ePipelineShaderStageCreateInfo,
                                                                .stage  = vk::ShaderStageFlagBits::eFragment,
                                                                .module = fragShaderModule,
                                                                .pName  = "main" };

        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{.vertexBindingDescriptionCount   =   1,
                                                               .pVertexBindingDescriptions      =   &bindingDescription, // Optional
                                                               .vertexAttributeDescriptionCount =   static_cast<uint32_t>(attributeDescriptions.size()),
                                                               .pVertexAttributeDescriptions    =   attributeDescriptions.data(), // Optional
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology               =   vk::PrimitiveTopology::eTriangleList,
                                                                .primitiveRestartEnable =   VK_FALSE };
        vk::Viewport viewport{  .x          =   0.0f,
                                .y          =   (float)swapChainExtent.height,
                                .width      =   (float)swapChainExtent.width,
                                .height     =   -(float)swapChainExtent.height,
                                .minDepth   =   0.0f,
                                .maxDepth   =   1.0f };
        vk::Rect2D scissor{ .offset =   {0,0},
                            .extent =   swapChainExtent };


        std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };

        vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                                                        .pDynamicStates = dynamicStates.data() };

        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1,
                                                            .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable           =   VK_FALSE,
                                                            .rasterizerDiscardEnable    =   VK_FALSE,
                                                            .polygonMode                =   vk::PolygonMode::eFill,
                                                            .cullMode                   =   vk::CullModeFlagBits::eBack,
                                                            .frontFace                  =   vk::FrontFace::eClockwise,
                                                            .depthBiasEnable            =   VK_FALSE,
                                                            .depthBiasConstantFactor    =   0.0f,   // Optional
                                                            .depthBiasClamp             =   0.0f,   // Optional
                                                            .depthBiasSlopeFactor       =   0.0f,   // Optional
                                                            .lineWidth                  =   1.0f };

        vk::PipelineMultisampleStateCreateInfo multisampling{   .rasterizationSamples   =   vk::SampleCountFlagBits::e1,
                                                                .sampleShadingEnable    =   VK_FALSE,
                                                                .minSampleShading       =   1.0f,         // Optional
                                                                .pSampleMask            =   nullptr,      // Optional
                                                                .alphaToCoverageEnable  =   VK_FALSE,     // Optional
                                                                .alphaToOneEnable       =   VK_FALSE };   // Optional
        
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable            = VK_FALSE,
                                                                    .srcColorBlendFactor    = vk::BlendFactor::eOne,    // Optional
                                                                    .dstColorBlendFactor    = vk::BlendFactor::eZero,   // Optional
                                                                    .colorBlendOp           = vk::BlendOp::eAdd,        // Optional
                                                                    .srcAlphaBlendFactor    = vk::BlendFactor::eOne,    // Optional
                                                                    .dstAlphaBlendFactor    = vk::BlendFactor::eZero,   // Optional
                                                                    .alphaBlendOp           = vk::BlendOp::eAdd,        // Optional
                                                                    .colorWriteMask         = vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };

        vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable  =   VK_FALSE,
                                                            .logicOp        =   vk::LogicOp::eCopy, // Optional
                                                            .attachmentCount=   1,
                                                            .pAttachments   =   &colorBlendAttachment,
            };
        colorBlending.setBlendConstants({ 0.f,0.f,0.f,0.f });

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount         =   1, // Optional
                                                        .pSetLayouts            =   &descriptorSetLayout, // Optional
                                                        .pushConstantRangeCount =   0, // Optional
                                                        .pPushConstantRanges    =   nullptr }; // Optional

        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{.stageCount         =   2,
                                                    .pStages            =   shaderStages,
                                                    .pVertexInputState  =   &vertexInputInfo,
                                                    .pInputAssemblyState=   &inputAssembly, 
                                                    .pViewportState     =   &viewportState,
                                                    .pRasterizationState=   &rasterizer,
                                                    .pMultisampleState  =   &multisampling,
                                                    .pDepthStencilState =   nullptr, // Optional
                                                    .pColorBlendState   =   &colorBlending,
                                                    .pDynamicState      =   &dynamicState,
                                                    .layout             =   pipelineLayout,
                                                    .renderPass         =   renderPass, 
                                                    .subpass            =   0,
                                                    .basePipelineHandle =   VK_NULL_HANDLE,
                                                    .basePipelineIndex  =   -1};    

        auto graphicsPipelineResult = device.createGraphicsPipeline(VK_NULL_HANDLE,pipelineInfo);  
        if (graphicsPipelineResult.result == vk::Result::eSuccess) {
            graphicsPipeline = graphicsPipelineResult.value;
        }
        else {
            throw std::runtime_error("failed to create graphics pipeline");
        }


        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);
    }

    vk::ShaderModule createShaderModule(const std::vector<char>& code) {
        vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size(),
                                                .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
        vk::ShaderModule shaderModule = device.createShaderModule(createInfo);
        return shaderModule;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] =createImageView(swapChainImages[i],swapChainImageFormat);
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo{  .surface            =   surface,
                                                .minImageCount      =   imageCount,
                                                .imageFormat        =   surfaceFormat.format,
                                                .imageColorSpace    =   surfaceFormat.colorSpace,
                                                .imageExtent        =   extent,
                                                .imageArrayLayers   =   1,
                                                .imageUsage         =   vk::ImageUsageFlagBits::eColorAttachment,
                                                .preTransform       =   swapChainSupport.capabilities.currentTransform,
                                                .compositeAlpha     =   vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                .presentMode        =   presentMode,
                                                .clipped            =   vk::True,
                                                .oldSwapchain       =   VK_NULL_HANDLE
        };
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        swapChain = device.createSwapchainKHR(createInfo);
        swapChainImages = device.getSwapchainImagesKHR(swapChain);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            vk::Extent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }


    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device) {
        SwapChainSupportDetails details{.capabilities   =   device.getSurfaceCapabilitiesKHR(surface),
                                        .formats        =   device.getSurfaceFormatsKHR(surface),
                                        .presentModes   =   device.getSurfacePresentModesKHR(surface)};



        return details;
    }

    void createSurface() {
        VkSurfaceKHR psurface;
        if (glfwCreateWindowSurface(instance, window, nullptr, &psurface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = psurface;
    }
            
    void pickPhysicalDevice() {
        auto devices = instance.enumeratePhysicalDevices();
        if (devices.size() == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        QueueFamilyIndices indicies = findQueueFamilies(physicalDevice);
        float queue_priority = 1.f;
        std::set<uint32_t> uniqueQueueFamilies = { indicies.graphicsFamily.value(), indicies.presentFamily.value() };
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo{  .queueFamilyIndex   =   queueFamily,
                                                        .queueCount         =   1,
                                                        .pQueuePriorities   =   &queue_priority };
            queueCreateInfos.push_back(queueCreateInfo);
        }



        vk::PhysicalDeviceFeatures deviceFeatures{.samplerAnisotropy = VK_TRUE };
        vk::DeviceCreateInfo createInfo{    .queueCreateInfoCount   =   static_cast<uint32_t>(queueCreateInfos.size()),
                                            .pQueueCreateInfos      =   queueCreateInfos.data(),
                                            .enabledExtensionCount  =   static_cast<uint32_t>(deviceExtensions.size()),
                                            .ppEnabledExtensionNames=  deviceExtensions.data(),
                                            .pEnabledFeatures       =   &deviceFeatures};

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }
        device = physicalDevice.createDevice(createInfo);
        graphicsQueue = device.getQueue(indicies.graphicsFamily.value(),0);
        presentQueue = device.getQueue(indicies.presentFamily.value(), 0);

    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device) {
        QueueFamilyIndices indices;
        auto queueFamilies = device.getQueueFamilyProperties2();
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }
            vk::Bool32 presentSupport = false;
            presentSupport = device.getSurfaceSupportKHR(i, surface);
            if (presentSupport) {
                indices.presentFamily = i;
            }
            if (indices.isComplete()) {
                break;  
            }
            i++;
        }
        return indices;
    }

    bool isDeviceSuitable(vk::PhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();

        return indices.isComplete() && extensionsSupported &&swapChainAdequate && supportedFeatures.samplerAnisotropy;
        
    }

    bool checkDeviceExtensionSupport(vk::PhysicalDevice device) {
        auto availableExtensions = device.enumerateDeviceExtensionProperties();

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {  .sType              =   vk::StructureType::eDebugUtilsMessengerCreateInfoEXT,
                        .messageSeverity    =   vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                        .messageType        =   vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                        .pfnUserCallback    =   debugCallback};
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
#pragma warning(suppress:4834) // (suppress one error in the next line)
        vk::enumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<vk::LayerProperties> availableLayers(layerCount);
#pragma warning(suppress:4834) // (suppress one error in the next line)
        vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        
        vk::DebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);
        createInfo.flags = vk::DebugUtilsMessengerCreateFlagsEXT();
        // NOTE: reinterpret_cast is also used by vulkan.hpp internally for all these structs
        if (CreateDebugUtilsMessengerEXT(
            instance,
            reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&createInfo),
            nullptr,
            &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up debug callback!");
        }
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }
        vk::ApplicationInfo appInfo{.pApplicationName   = "Hello Vulkan!",
                                    .applicationVersion = vk::makeApiVersion(0,1,0,0),
                                    .pEngineName        = "No engine",
                                    .engineVersion      = vk::makeApiVersion(0,1,0,0),
                                    .apiVersion         = vk::makeVersion(1,3,0) };

        vk::InstanceCreateInfo createInfo{.pApplicationInfo   = &appInfo };

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo;

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (vk::DebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }
        
        vk::Result result = vk::createInstance(&createInfo, nullptr, &instance);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to create an instance!");
        }

    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }


        device.waitIdle();
    }

};

int main() {    
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}