# Build a static library with our common helper/verification code
add_library(common STATIC
        src/verification.cpp
)

# Make the "include" directory available to other targets that link to this
target_include_directories(common PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)