# VLRF

My LLM side project to make use of old cpus and remove external dependencies.
All of it was created under codename so the repository does not contain the original structure of commits.

# Build

`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j`
For general basic features

To use OpenCL and SYCL you need to install required dependencies and add required flags.