# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2026 Aleksander "Kowalski"
cmake_minimum_required(VERSION 3.20)

if(MSVC)
    string(REPLACE "/Ob2" "/Ob3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Ob2" "/Ob3" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
endif()

project(VLRF VERSION 1.1.2 LANGUAGES CXX)

# =============================================================================
# C++ standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Options
# =============================================================================
option(VLRF_BUILD_TESTS    "Build unit tests"           OFF)
option(VLRF_BUILD_TOOLS    "Build CLI tools"            ON)
option(VLRF_USE_OPENCL     "Enable OpenCL backend"      OFF)
option(VLRF_USE_SYCL       "Enable Intel SYCL backend"  OFF)
option(VLRF_USE_ONEMKL     "Use Intel oneMKL for GEMM"  OFF)
option(VLRF_ENABLE_LTO     "Enable link-time optimisation" OFF)
option(VLRF_USE_AVX2       "Enable hand-written AVX2/FMA GEMM kernels (ignored if VLRF_USE_ONEMKL=ON)" OFF)

# Coding-agent sandbox backends (vlrf_cli agent). Default ON on their native
# platform, OFF cross-platform, since each depends on host-specific
# tooling (Podman CLI on the PATH; Hyper-V + Windows SDK's computecore.h
# on Windows). "auto" backend selection at runtime (see AgentConfig::
# resolve_backend in include/agent/agent_config.hpp) is a NO-OP unless the
# matching option here was ON at configure time — resolve_backend()
# throws a clear error rather than silently falling back if the request
# doesn't match what this build was compiled with.
option(VLRF_AGENT_USE_PODMAN "Enable the Podman sandbox backend for the coding agent (Linux/macOS)" OFF)
option(VLRF_AGENT_USE_HYPERV "Enable the Hyper-V sandbox backend for the coding agent (Windows only, requires Windows SDK's computecore.h)" OFF)

# Explicit SIMD target — overrides VLRF_USE_AVX2 when set to AVX1.
# Use AVX1 for laptops/older machines: 3rd-gen Intel Core (Ivy Bridge, 2012)
# and 2nd-gen (Sandy Bridge, 2011) have 256-bit AVX float ops but no FMA3,
# which only arrived with 4th-gen Haswell (2013). Building with AVX2/FMA
# flags on these CPUs causes "illegal instruction" crashes even with our
# runtime detection, because /arch:AVX2 affects the WHOLE compilation unit,
# not just the explicit SIMD intrinics calls.
#
#   AUTO  - AVX2 if VLRF_USE_AVX2=ON, else AVX1            (default, desktops)
#   AVX2  - force AVX2+FMA compilation (Haswell 2013+)
#   AVX1  - force AVX1-only compilation, NO FMA           (Sandy/Ivy Bridge, ARM laptops with old emulation, etc.)
#   NONE  - no explicit /arch flag, safest, slowest
set(VLRF_TARGET_ARCH "AUTO" CACHE STRING
    "SIMD target: AUTO, AVX2, AVX1, or NONE")
set_property(CACHE VLRF_TARGET_ARCH PROPERTY STRINGS AUTO AVX2 AVX1 NONE)

# =============================================================================
# Platform detection
# =============================================================================
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Fail fast with a clear message rather than a wall of missing-header
# errors deep in sandbox_hyperv.cpp if someone forces this ON cross-platform.
if(VLRF_AGENT_USE_HYPERV AND NOT WIN32)
    message(FATAL_ERROR
        "VLRF_AGENT_USE_HYPERV=ON requires Windows (Host Compute Service API, "
        "computecore.h, is Windows-only). Use VLRF_AGENT_USE_PODMAN on Linux/macOS instead.")
endif()

# =============================================================================
# Compiler flags
# =============================================================================

if(MSVC)
    # /arch:AVX2 is REQUIRED for the hand-written SIMD GEMM kernels in
    # simd_gemm.hpp (_mm256_fmadd_ps and friends need AVX2+FMA3).
    # /arch:AVX alone does NOT enable FMA on MSVC and silently falls back
    # to slower multiply+add sequences, or may not compile FMA intrinsics
    # at all depending on the MSVC version.
    #
    # IMPORTANT: /arch:AVX2 applies to the WHOLE translation unit, not just
    # the explicit intrinsic calls. The runtime cpu_features() check in
    # simd_gemm.hpp only guards our hand-written kernels; it cannot prevent
    # the compiler from auto-vectorising OTHER loops with AVX2 instructions.
    # This binary will crash with "illegal instruction" on a CPU without
    # AVX2 support (anything older than ~2013 Intel Haswell / AMD Excavator).
    # If you need to support pre-AVX2 CPUs, build a second binary with
    # /arch:AVX (or no /arch flag) and ship both, selecting at install time.
    add_compile_options(/W3 /openmp)
    add_compile_definitions(_USE_MATH_DEFINES)
    if(NOT VLRF_USE_ONEMKL)
        if(VLRF_TARGET_ARCH STREQUAL "AVX2")
            add_compile_options(/arch:AVX2)
            message(STATUS "  SIMD target    : AVX2+FMA (explicit)")
        elseif(VLRF_TARGET_ARCH STREQUAL "AVX1")
            add_compile_options(/arch:AVX)
            message(STATUS "  SIMD target    : AVX1 only, no FMA (explicit — Sandy/Ivy Bridge)")
        elseif(VLRF_TARGET_ARCH STREQUAL "NONE")
            message(STATUS "  SIMD target    : none (no /arch flag — safest, slowest)")
        elseif(VLRF_USE_AVX2)
            add_compile_options(/arch:AVX2)
            message(STATUS "  SIMD target    : AVX2+FMA (auto, VLRF_USE_AVX2=ON)")
        else()
            add_compile_options(/arch:AVX)
            message(STATUS "  SIMD target    : AVX1 only, no FMA (auto, VLRF_USE_AVX2=OFF)")
        endif()
    endif()

    add_compile_options($<$<CONFIG:Release>:/O2>)
    add_compile_options($<$<CONFIG:Release>:/Ob3>)      # aggressive inlining
    add_compile_options($<$<CONFIG:Release>:/GL>)       # whole-program optimisation
    add_link_options($<$<CONFIG:Release>:/LTCG>)        # link-time code generation
    add_compile_options($<$<CONFIG:Release>:/DNDEBUG>)  # disable asserts
    # add_compile_definitions($<$<CONFIG:Release>:NDEBUG>)

else()
    # GCC/Clang
    add_compile_options(
        -Wall -Wextra -O3
        -funroll-loops
        -ffunction-sections
        -fdata-sections
    )
    if(NOT VLRF_USE_ONEMKL)
        if(VLRF_TARGET_ARCH STREQUAL "AVX2")
            add_compile_options(-mavx2 -mfma)
            message(STATUS "  SIMD target    : AVX2+FMA (explicit)")
        elseif(VLRF_TARGET_ARCH STREQUAL "AVX1")
            add_compile_options(-mavx -mno-fma)
            message(STATUS "  SIMD target    : AVX1 only, no FMA (explicit — Sandy/Ivy Bridge)")
        elseif(VLRF_TARGET_ARCH STREQUAL "NONE")
            message(STATUS "  SIMD target    : none (portable, slowest)")
        elseif(VLRF_USE_AVX2)
            add_compile_options(-mavx2 -mfma)
            message(STATUS "  SIMD target    : AVX2+FMA (auto, VLRF_USE_AVX2=ON)")
        else()
            add_compile_options(-mavx -mno-fma)
            message(STATUS "  SIMD target    : AVX1 only, no FMA (auto, VLRF_USE_AVX2=OFF)")
        endif()
    endif()

    if(APPLE)
        add_link_options($<$<CONFIG:Release>:-flto>)
        add_link_options($<$<CONFIG:Release>:-Wl,-dead_strip>)
    else()
        add_link_options($<$<CONFIG:Release>:-flto>)
        add_link_options($<$<CONFIG:Release>:-Wl,--gc-sections>)
    endif()

    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        link_libraries(OpenMP::OpenMP_CXX)
    endif()
endif()


# =============================================================================
# Optional backends
# =============================================================================
if(VLRF_USE_OPENCL)
    find_package(OpenCL REQUIRED)
    add_compile_definitions(VLRF_USE_OPENCL)
endif()

if(VLRF_USE_SYCL)
    if(CMAKE_CXX_COMPILER MATCHES "(icx|icpx|icx-cl|dpcpp)(\\.exe)?$" OR
       CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
        add_compile_options(-fsycl)
        add_link_options(-fsycl)
        add_compile_definitions(_HAS_CXX23=0)
        set(_sycl_available ON)
    endif()

    if(NOT _sycl_available)
        message(WARNING "VLRF_USE_SYCL=ON but no usable Intel SYCL toolchain was detected...")
        set(VLRF_USE_SYCL OFF)
    else()
        add_compile_definitions(VLRF_USE_SYCL)
        message(STATUS "  SYCL toolchain  : enabled (manual flags, bypassing IntelSYCLConfig)")
    endif()
endif()

if(VLRF_USE_ONEMKL)
    find_package(MKL REQUIRED)
    add_compile_definitions(VLRF_USE_ONEMKL)
endif()

# =============================================================================
# SYCL runtime DLLs (Windows) — auto-copy next to build output
# =============================================================================
# icx-cl/icx/dpcpp don't statically link the SYCL runtime; targets built with
# -fsycl need sycl*.dll (+ svml_dispmd.dll, libmmd.dll, and a few others) to
# be discoverable at process launch. On a machine with a full oneAPI command
# prompt / PATH setup this is already handled, but a plain `Run` from
# Visual Studio / VS Code, or double-clicking the exe in Explorer, won't have
# that PATH — hence "builds fine, DLL not found at launch".
#
# VLRF_SYCL_RUNTIME_DIR is derived from CMAKE_CXX_COMPILER's own location
# (<oneAPI compiler root>\bin, sibling of icx-cl.exe/icx.exe) rather than a
# hardcoded version path, so this keeps working across oneAPI versions and
# different install locations.
if(WIN32 AND VLRF_USE_SYCL AND _sycl_available)
    get_filename_component(VLRF_SYCL_RUNTIME_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)

    # Every SYCL runtime DLL vlrf has been observed to need at launch.
    # If a future oneAPI/driver update reports a different missing DLL,
    # add its name here — this list is not derived from anything
    # machine-readable, since oneAPI doesn't expose one.
    set(VLRF_SYCL_RUNTIME_DLLS
        sycl9.dll        sycl9d.dll        # SYCL runtime (release/debug)
        sycl8.dll         sycl8d.dll        # older oneAPI versions ship these instead
        sycl7.dll         sycl7d.dll        # ...and older still
        svml_dispmd.dll                     # Intel Short Vector Math Library
        libmmd.dll        libmmdd.dll       # Intel math/CRT support (release/debug)
        libiomp5md.dll                      # Intel OpenMP runtime (used alongside /openmp)
        pi_win_proxy_loader.dll             # SYCL plugin-interface proxy loader
        pi_level_zero.dll                   # Level Zero backend plugin
        pi_opencl.dll                       # OpenCL backend plugin
        ur_loader.dll                       # Unified Runtime loader (newer oneAPI)
        ur_adapter_level_zero.dll
        ur_adapter_opencl.dll
        ur_win_proxy_loader.dll
    )

    function(vlrf_copy_sycl_runtime_dlls target)
        foreach(dll_name IN LISTS VLRF_SYCL_RUNTIME_DLLS)
            set(dll_src "${VLRF_SYCL_RUNTIME_DIR}/${dll_name}")
            # Not every name in the list exists in every oneAPI version —
            # only queue a copy for the ones actually present, so the build
            # doesn't fail trying to copy a nonexistent file.
            if(EXISTS "${dll_src}")
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll_src}"
                        "$<TARGET_FILE_DIR:${target}>"
                    COMMENT "Copying ${dll_name} next to ${target}"
                    VERBATIM
                )
            endif()
        endforeach()
    endfunction()
else()
    function(vlrf_copy_sycl_runtime_dlls target)
        # No-op when SYCL isn't enabled — keeps call sites below unconditional.
    endfunction()
endif()

# =============================================================================
# Include paths
# =============================================================================
set(VLRF_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/model
    ${CMAKE_CURRENT_SOURCE_DIR}/include/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/include/api
    ${CMAKE_CURRENT_SOURCE_DIR}/include/agent
)

# =============================================================================
# VLRF shared library (DLL on Windows)
# =============================================================================
add_library(vlrf SHARED
    src/api/vlrf_api.cpp
)

target_include_directories(vlrf PRIVATE ${VLRF_INCLUDE_DIRS})
target_compile_definitions(vlrf PRIVATE VLRF_BUILD_DLL)

if(VLRF_USE_OPENCL)
    target_link_libraries(vlrf PRIVATE OpenCL::OpenCL)
endif()
if(VLRF_USE_SYCL)
    if(TARGET SYCL::SYCL)
        target_link_libraries(vlrf PRIVATE SYCL::SYCL)
    elseif(TARGET IntelSYCL::SYCL)
        target_link_libraries(vlrf PRIVATE IntelSYCL::SYCL)
    endif()
endif()
if(VLRF_USE_ONEMKL)
    target_link_libraries(vlrf PRIVATE MKL::MKL)
endif()

# Threads (for std::thread in ops.hpp)
find_package(Threads REQUIRED)
target_link_libraries(vlrf PRIVATE Threads::Threads)

# Windows-specific: generate import library alongside DLL
if(WIN32)
    set_target_properties(vlrf PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF  # we use explicit __declspec
        PREFIX ""
        SUFFIX ".dll"
    )
endif()

vlrf_copy_sycl_runtime_dlls(vlrf)

# =============================================================================
# VLRF static library (for linking into tools / tests without DLL)
# =============================================================================
add_library(vlrf_static STATIC
    src/api/vlrf_api.cpp
)
target_include_directories(vlrf_static PRIVATE ${VLRF_INCLUDE_DIRS})
target_compile_definitions(vlrf_static PRIVATE VLRF_BUILD_DLL)

if(VLRF_USE_OPENCL)
    target_compile_definitions(vlrf_static PRIVATE VLRF_USE_OPENCL)
    target_link_libraries(vlrf_static PUBLIC OpenCL::OpenCL)
endif()

if(VLRF_USE_SYCL)
    target_compile_definitions(vlrf_static PRIVATE VLRF_USE_SYCL)
    if(TARGET SYCL::SYCL)
        target_link_libraries(vlrf_static PUBLIC SYCL::SYCL)
    elseif(TARGET IntelSYCL::SYCL)
        target_link_libraries(vlrf_static PUBLIC IntelSYCL::SYCL)
    endif()
endif()

if(VLRF_USE_ONEMKL)
    target_compile_definitions(vlrf_static PRIVATE VLRF_USE_ONEMKL)
    target_link_libraries(vlrf_static PUBLIC MKL::MKL)
endif()

target_link_libraries(vlrf_static PRIVATE Threads::Threads)

# =============================================================================
# VLRF agent library (coding-agent loop + sandbox backends)
#
# Kept as its own static library rather than folded into vlrf_static:
# it depends on neither the transformer/backend code nor VLRF_BUILD_DLL, and
# keeping it separate means tools/tests that don't need the agent loop
# (e.g. the existing test_*.cpp binaries) don't pull in Hyper-V/Podman
# platform code or its extra link libraries at all.
# =============================================================================
add_library(vlrf_agent STATIC
    src/agent/tiny_json.cpp
    src/agent/agent_config.cpp
    src/agent/tool_protocol.cpp
    src/agent/sandbox.cpp
    src/agent/sandbox_factory.cpp
    src/agent/agent_loop.cpp
)
target_include_directories(vlrf_agent PUBLIC ${VLRF_INCLUDE_DIRS})
target_link_libraries(vlrf_agent PRIVATE Threads::Threads)

if(VLRF_AGENT_USE_PODMAN)
    target_sources(vlrf_agent PRIVATE src/agent/sandbox_podman.cpp)
    target_compile_definitions(vlrf_agent PUBLIC VLRF_AGENT_USE_PODMAN)
endif()

if(VLRF_AGENT_USE_HYPERV)
    target_sources(vlrf_agent PRIVATE src/agent/sandbox_hyperv.cpp)
    target_compile_definitions(vlrf_agent PUBLIC VLRF_AGENT_USE_HYPERV)
    # computecore.lib / shlwapi.lib ship with the Windows SDK (no separate
    # NuGet/vcpkg package needed) — just need the SDK's lib dir on the
    # linker path, which CMake's default Visual Studio generator already
    # sets up. computecore.h itself needs a reasonably recent SDK (the
    # HCS v2 schema targeted here matches Windows 11 24H2 / Server 2022+
    # headers — see the schema-version comment in sandbox_hyperv.cpp).
    target_link_libraries(vlrf_agent PRIVATE computecore.lib shlwapi.lib)
endif()

if(NOT VLRF_AGENT_USE_PODMAN AND NOT VLRF_AGENT_USE_HYPERV)
    message(STATUS
        "Neither VLRF_AGENT_USE_PODMAN nor VLRF_AGENT_USE_HYPERV is enabled: "
        "'vlrf_cli agent' will build but can only run with sandbox_backend=none "
        "(i.e. it cannot actually execute tool calls). Enable the backend "
        "matching your platform to use the coding-agent loop for real.")
endif()

# =============================================================================
# CLI Tool: vlrf_cli
# =============================================================================
if(VLRF_BUILD_TOOLS)
    add_executable(vlrf_cli tools/vlrf_cli.cpp)
    target_include_directories(vlrf_cli PRIVATE ${VLRF_INCLUDE_DIRS})
    target_compile_definitions(vlrf_cli PRIVATE VLRF_BUILD_DLL)
    if(VLRF_USE_OPENCL)
        target_compile_definitions(vlrf_cli PRIVATE VLRF_USE_OPENCL)
    endif()
    if(VLRF_USE_SYCL)
        target_compile_definitions(vlrf_cli PRIVATE VLRF_USE_SYCL)
    endif()
    target_link_libraries(vlrf_cli PRIVATE vlrf_static vlrf_agent Threads::Threads)
    if(WIN32)
        target_compile_definitions(vlrf_cli PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    vlrf_copy_sycl_runtime_dlls(vlrf_cli)
endif()

# =============================================================================
# Link-time optimisation (must come after the targets are defined)
# =============================================================================
if(VLRF_ENABLE_LTO AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    cmake_policy(SET CMP0069 NEW)
    foreach(tgt vlrf vlrf_static vlrf_agent vlrf_cli)
        if(TARGET ${tgt})
            set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    endforeach()
endif()

# =============================================================================
# Tests
# =============================================================================
if(VLRF_BUILD_TESTS)
    enable_testing()

    add_executable(test_tensor   tests/test_tensor.cpp)
    add_executable(test_ops      tests/test_ops.cpp)
    add_executable(test_model    tests/test_model.cpp)
    add_executable(test_api      tests/test_api.cpp)
    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    add_executable(test_gpt2_pretokenizer tests/test_gpt2_pretokenizer.cpp)
    add_executable(test_kv_cache_quant tests/test_kv_cache_quant.cpp)
    add_executable(test_flash_attention tests/test_flash_attention.cpp)
    add_executable(test_tokenizer_fixes tests/test_tokenizer_fixes.cpp)
    if(VLRF_AGENT_USE_HYPERV)
        # Exercises the New-VM/Start-VM/Invoke-Command/Copy-VMFile sequence used by
        # sandbox_hyperv.cpp. Needs no extra libs (shlwapi/user32 via #pragma comment).
        add_executable(newvm_probe tests/newvm_probe.cpp)
    endif()

    foreach(tgt
        test_tensor test_ops test_model test_api test_tokenizer test_gpt2_pretokenizer test_kv_cache_quant test_flash_attention test_tokenizer_fixes)
        target_include_directories(${tgt} PRIVATE ${VLRF_INCLUDE_DIRS})
        if(VLRF_USE_OPENCL)
            target_compile_definitions(${tgt} PRIVATE VLRF_USE_OPENCL)
        endif()
        if(VLRF_USE_SYCL)
            target_compile_definitions(${tgt} PRIVATE VLRF_USE_SYCL)
        endif()
        target_link_libraries(${tgt} PRIVATE vlrf_static Threads::Threads)
        add_test(NAME ${tgt} COMMAND ${tgt})
        vlrf_copy_sycl_runtime_dlls(${tgt})
    endforeach()

    # Manual tokenizer inspection tools: they need a tokenizer.nft path on the
    # command line, so they are built but deliberately NOT registered with
    # add_test (ctest would run them with no arguments and they'd fail).
    foreach(tgt test_real_tokenizer_encode test_roundtrip)
        add_executable(${tgt} tests/${tgt}.cpp)
        target_include_directories(${tgt} PRIVATE ${VLRF_INCLUDE_DIRS})
        target_link_libraries(${tgt} PRIVATE vlrf_static Threads::Threads)
        vlrf_copy_sycl_runtime_dlls(${tgt})
    endforeach()
endif()

# =============================================================================
# Install rules
# =============================================================================
install(TARGETS vlrf vlrf_static
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES include/api/vlrf.h DESTINATION include/vlrf)
install(FILES LICENSE.txt DESTINATION share/doc/vlrf)
install(DIRECTORY include/model include/backend
        DESTINATION include/vlrf)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "=== VLRF Build Configuration ===")
message(STATUS "  AVX2/FMA GEMM  : ${VLRF_USE_AVX2}")
message(STATUS "  OpenCL backend : ${VLRF_USE_OPENCL}")
message(STATUS "  SYCL backend   : ${VLRF_USE_SYCL}")
message(STATUS "  oneMKL GEMM    : ${VLRF_USE_ONEMKL}")
message(STATUS "  Build tests    : ${VLRF_BUILD_TESTS}")
message(STATUS "  Build tools    : ${VLRF_BUILD_TOOLS}")
message(STATUS "  Agent (Podman) : ${VLRF_AGENT_USE_PODMAN}")
message(STATUS "  Agent (Hyper-V): ${VLRF_AGENT_USE_HYPERV}")
message(STATUS "======================================")
