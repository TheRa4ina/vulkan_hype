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

#move textures 
add_custom_command(TARGET AnchorApp POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/textures ${CMAKE_BINARY_DIR}/textures)