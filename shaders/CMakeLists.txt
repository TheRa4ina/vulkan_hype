# shaders
set(shaders_full_path ${CMAKE_CURRENT_LIST_DIR})
set(shaders_path "shaders")
# file(GLOB shaders RELATIVE ${shaders_full_path} *.vert *.frag)
set(shaders 
"shader.frag"
"shader.vert"
)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR})



message("shaders path searching at "${shaders_full_path})
foreach(shader ${shaders})
set(input_glsl ${CMAKE_CURRENT_LIST_DIR}/${shader})
set(output_spv ${CMAKE_BINARY_DIR}/${shaders_path}/${shader}.spv)
message(${shader} " is compiled to " ${output_spv})
execute_process(COMMAND "glslc" "${input_glsl}" "-o" "${output_spv}")
endforeach()