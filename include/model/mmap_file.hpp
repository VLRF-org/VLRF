// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — MmapFile  (mmap_file.hpp)
// Cross-platform memory-mapped file.
// On Windows: CreateFileMapping / MapViewOfFile
// On Linux:   mmap(MAP_SHARED | MAP_POPULATE)
//
// The mapped region is read-only. Tensors that point into it have
// TensorStorage::owned = false so they never try to free it.
// The MmapFile object must outlive all tensors that reference it.
// =============================================================================
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef _WIN32
// #  define WIN32_LEAN_AND_MEAN
// #  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace vlrf {

class MmapFile {
public:
    const uint8_t* data {nullptr};
    size_t         size {0};

    MmapFile() = default;

    explicit MmapFile(std::string const& path) { open(path); }

    ~MmapFile() { close(); }

    // Non-copyable, movable
    MmapFile(MmapFile const&)            = delete;
    MmapFile& operator=(MmapFile const&) = delete;

    MmapFile(MmapFile&& o) noexcept {
        data   = o.data;   size   = o.size;
#ifdef _WIN32
        hfile_ = o.hfile_; hmap_  = o.hmap_;
        o.hfile_ = INVALID_HANDLE_VALUE;
        o.hmap_  = nullptr;
#else
        fd_    = o.fd_;
        o.fd_  = -1;
#endif
        o.data = nullptr;  o.size = 0;
    }

    void open(std::string const& path) {
        close();

#ifdef _WIN32
        hfile_ = CreateFileA(path.c_str(),
                             GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hfile_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("MmapFile: cannot open '" + path + "'");

        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hfile_, &sz)) {
            CloseHandle(hfile_); hfile_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("MmapFile: GetFileSizeEx failed");
        }
        size = static_cast<size_t>(sz.QuadPart);

        hmap_ = CreateFileMappingA(hfile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hmap_) {
            CloseHandle(hfile_); hfile_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("MmapFile: CreateFileMapping failed");
        }

        void* ptr = MapViewOfFile(hmap_, FILE_MAP_READ, 0, 0, 0);
        if (!ptr) {
            CloseHandle(hmap_); CloseHandle(hfile_);
            hmap_ = nullptr; hfile_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("MmapFile: MapViewOfFile failed");
        }
        data = static_cast<const uint8_t*>(ptr);

#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("MmapFile: cannot open '" + path + "'");

        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("MmapFile: fstat failed");
        }
        size = static_cast<size_t>(st.st_size);

        void* ptr = ::mmap(nullptr, size, PROT_READ,
                           MAP_SHARED | MAP_POPULATE, fd_, 0);
        if (ptr == MAP_FAILED) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("MmapFile: mmap failed");
        }
        data = static_cast<const uint8_t*>(ptr);
#endif
    }

    void close() {
        if (!data) return;
#ifdef _WIN32
        UnmapViewOfFile(data);
        if (hmap_  && hmap_  != INVALID_HANDLE_VALUE) CloseHandle(hmap_);
        if (hfile_ && hfile_ != INVALID_HANDLE_VALUE) CloseHandle(hfile_);
        hmap_ = nullptr; hfile_ = INVALID_HANDLE_VALUE;
#else
        if (size) ::munmap(const_cast<uint8_t*>(data), size);
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        data = nullptr; size = 0;
    }

    bool is_open() const { return data != nullptr; }

    // Typed pointer into the mapped region
    template<typename T>
    const T* ptr_at(size_t byte_offset) const {
        return reinterpret_cast<const T*>(data + byte_offset);
    }

    // Read a value at offset (handles unaligned reads safely)
    template<typename T>
    T read_at(size_t byte_offset) const {
        T v;
        __builtin_memcpy(&v, data + byte_offset, sizeof(T));
        return v;
    }

    // Advise the OS about access patterns (best-effort, no-op if unsupported)
    void advise_sequential(size_t offset, size_t len) const {
#if !defined(_WIN32) && defined(MADV_SEQUENTIAL)
        ::madvise(const_cast<uint8_t*>(data) + offset, len, MADV_SEQUENTIAL);
#else
        (void)offset; (void)len;
#endif
    }

    void advise_random(size_t offset, size_t len) const {
#if !defined(_WIN32) && defined(MADV_RANDOM)
        ::madvise(const_cast<uint8_t*>(data) + offset, len, MADV_RANDOM);
#else
        (void)offset; (void)len;
#endif
    }

    void advise_willneed(size_t offset, size_t len) const {
#if !defined(_WIN32) && defined(MADV_WILLNEED)
        ::madvise(const_cast<uint8_t*>(data) + offset, len, MADV_WILLNEED);
#else
        (void)offset; (void)len;
#endif
    }

    void advise_dontneed(size_t offset, size_t len) const {
#if !defined(_WIN32) && defined(MADV_DONTNEED)
        ::madvise(const_cast<uint8_t*>(data) + offset, len, MADV_DONTNEED);
#else
        (void)offset; (void)len;
#endif
    }

private:
#ifdef _WIN32
    HANDLE hfile_ {INVALID_HANDLE_VALUE};
    HANDLE hmap_  {nullptr};
#else
    int fd_ {-1};
#endif
};

} // namespace vlrf
