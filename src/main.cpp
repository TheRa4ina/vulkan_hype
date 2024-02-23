#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
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
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const std::vector<const char*> validationLayers = {
"VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
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
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffer();
        createSyncObjects();
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
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            device.destroySemaphore(imageAvailableSemaphores[i]);
            device.destroySemaphore(renderFinishedSemaphores[i]);
            device.destroyFence(inFlightFences[i]);
        }
        cleanupSwapChain();
        device.destroyCommandPool(commandPool);
        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);
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

        vk::Viewport viewport{  .x          =   0.0f,
                                .y          =   0.0f,
                                .width      =   static_cast<float>(swapChainExtent.width),
                                .height     =   static_cast<float>(swapChainExtent.height),
                                .minDepth   =   0.0f,
                                .maxDepth   =   1.0f };
        commandBuffer.setViewport(0,viewport);

        vk::Rect2D scissor{ .offset =   { 0, 0 },
                            .extent =   swapChainExtent };
        commandBuffer.setScissor(0, scissor);
        commandBuffer.draw(3, 1, 0, 0);
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
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ .vertexBindingDescriptionCount  =   0,
                                                               .pVertexBindingDescriptions      =   nullptr, // Optional
                                                               .vertexAttributeDescriptionCount =   0,
                                                               .pVertexAttributeDescriptions    =   nullptr, // Optional
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology               =   vk::PrimitiveTopology::eTriangleList,
                                                                .primitiveRestartEnable =   VK_FALSE };
        vk::Viewport viewport{  .x          =   0.0f,
                                .y          =   0.0f,
                                .width      =   (float)swapChainExtent.width,
                                .height     =   (float)swapChainExtent.height,
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
                                                                    .srcColorBlendFactor    = vk::BlendFactor::eOne,   // Optional
                                                                    .dstColorBlendFactor    = vk::BlendFactor::eZero,  // Optional
                                                                    .colorBlendOp           = vk::BlendOp::eAdd,              // Optional
                                                                    .srcAlphaBlendFactor    = vk::BlendFactor::eOne,   // Optional
                                                                    .dstAlphaBlendFactor    = vk::BlendFactor::eZero,  // Optional
                                                                    .alphaBlendOp           = vk::BlendOp::eAdd,              // Optional
                                                                    .colorWriteMask         = vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };

        vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable  =   VK_FALSE,
                                                            .logicOp        =   vk::LogicOp::eCopy, // Optional
                                                            .attachmentCount=   1,
                                                            .pAttachments   =   &colorBlendAttachment,
            };
        colorBlending.setBlendConstants({ 0.f,0.f,0.f,0.f });

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount         =   0, // Optional
                                                        .pSetLayouts            =   nullptr, // Optional
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
            vk::ImageViewCreateInfo createInfo{ .image          =   swapChainImages[i],
                                                .viewType       =   vk::ImageViewType::e2D,
                                                .format         =   swapChainImageFormat};
            createInfo.components.r = vk::ComponentSwizzle::eIdentity;
            createInfo.components.g = vk::ComponentSwizzle::eIdentity;
            createInfo.components.b = vk::ComponentSwizzle::eIdentity;
            createInfo.components.a = vk::ComponentSwizzle::eIdentity;
            createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            swapChainImageViews[i] = device.createImageView(createInfo);
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
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
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
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        float queue_priority = 1.f;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo{  .queueFamilyIndex   =   queueFamily,
                                                        .queueCount         =   1,
                                                        .pQueuePriorities   =   &queue_priority };
            queueCreateInfos.push_back(queueCreateInfo);
        }



        vk::PhysicalDeviceFeatures deviceFeatures{};
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
        graphicsQueue = device.getQueue(indices.graphicsFamily.value(),0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);

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

        return indices.isComplete() && extensionsSupported &&swapChainAdequate;
        
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
                                    .apiVersion         = vk::makeVersion(1,1,0) };

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