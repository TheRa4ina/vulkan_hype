#src

add_executable(AnchorApp
	main.cpp
)

target_include_directories(AnchorApp
PRIVATE 
	${CMAKE_CURRENT_LIST_DIR}/..
)
target_link_libraries(AnchorApp 
PRIVATE 
	Hello
	Vulkan::Vulkan
	Vulkan::Headers
	glfw
	glm::glm
)	
