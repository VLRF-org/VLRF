# VLRF

My LLM side project to make use of old cpus and remove external dependencies.
This repository is a downstream copy of my main repository. It was created under a codename, so the original commit history and repository structure are not preserved.

# Build

`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j`
For general basic features

To use OpenCL and SYCL you need to install required dependencies and add required flags.

For more context I got results of 7 tok/s on 14 year old cpu (light model ~1b parameters) and this was main goal.