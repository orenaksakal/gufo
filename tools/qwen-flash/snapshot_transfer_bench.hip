// Frozen UMA snapshot transfer: fresh pageable destinations, retained peers.
// Build: tools/bench/build.sh tools/qwen-flash/snapshot_transfer_bench.hip
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}

static double Ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

static std::uint64_t Compactions() {
  std::ifstream input("/proc/vmstat");
  std::string key;
  std::uint64_t value;
  while (input >> key >> value)
    if (key == "compact_stall")
      return value;
  throw std::runtime_error("cannot read host compaction counter");
}

static void Prepare(unsigned char* data, std::size_t bytes, bool huge,
                    unsigned workers) {
  const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const auto skip = (page - address % page) % page;
  const auto count = (bytes - skip) / page * page;
  if (madvise(data + skip, count, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE))
    throw std::runtime_error("snapshot page advice failed");
  if (!workers)
    return;
  constexpr std::size_t chunk = 16ULL << 20;
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  auto populate = [&] {
    for (;;) {
      const auto offset = next.fetch_add(chunk, std::memory_order_relaxed);
      if (offset >= count)
        return;
      if (madvise(data + skip + offset, std::min(chunk, count - offset),
                  MADV_POPULATE_WRITE))
        failed.store(true, std::memory_order_relaxed);
    }
  };
  {
    std::vector<std::jthread> threads;
    for (unsigned i = 1; i < workers; ++i)
      threads.emplace_back(populate);
    populate();
  }
  if (failed)
    throw std::runtime_error("snapshot page population failed");
}

int main(int argc, char** argv) {
  const std::size_t bytes = (argc > 1 ? std::stoull(argv[1]) : 1024) << 20;
  if (bytes < (16ULL << 20))
    return 2;
  void* source = nullptr;
  Check(hipMalloc(&source, bytes));
  Check(hipMemset(source, 0x5a, bytes));
  Check(hipDeviceSynchronize());
  hipStream_t stream;
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  std::vector<std::unique_ptr<unsigned char[]>> retained;
  for (unsigned i = 0; i < 2; ++i) {
    retained.emplace_back(new unsigned char[bytes]);
    Prepare(retained.back().get(), bytes, true, 4);
    std::memset(retained.back().get(), 0x37, bytes);
  }
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    for (const auto huge : {true, false}) {
      for (const unsigned workers : {0, 1, 4, 8}) {
        auto destination =
            std::make_unique_for_overwrite<unsigned char[]>(bytes);
        rusage before{}, after{};
        getrusage(RUSAGE_SELF, &before);
        const auto compactions = Compactions();
        const auto start = std::chrono::steady_clock::now();
        Prepare(destination.get(), bytes, huge, workers);
        const auto prepare_ms = Ms(start);
        const auto copy = std::chrono::steady_clock::now();
        constexpr std::size_t region = 128ULL << 20;
        for (std::size_t off = 0; off < bytes; off += region) {
          Check(hipMemcpyAsync(destination.get() + off,
                               static_cast<unsigned char*>(source) + off,
                               std::min(region, bytes - off),
                               hipMemcpyDeviceToHost, stream));
          Check(hipStreamSynchronize(stream));
        }
        const auto copy_ms = Ms(copy);
        getrusage(RUSAGE_SELF, &after);
        const bool exact =
            std::all_of(destination.get(), destination.get() + bytes,
                        [](unsigned char x) { return x == 0x5a; });
        const auto release = std::chrono::steady_clock::now();
        destination.reset();
        const auto release_ms = Ms(release);
        std::printf(
            "bytes=%zu repeat=%u huge=%d workers=%u prepare_ms=%.3f "
            "copy_ms=%.3f total_ms=%.3f release_ms=%.3f cycle_ms=%.3f "
            "faults=%ld compactions=%llu exact=%d\n",
            bytes, repeat, huge, workers, prepare_ms, copy_ms,
            prepare_ms + copy_ms, release_ms, prepare_ms + copy_ms + release_ms,
            after.ru_minflt - before.ru_minflt,
            static_cast<unsigned long long>(Compactions() - compactions),
            exact);
        std::fflush(stdout);
        if (!exact)
          return 1;
      }
    }
  }
  Check(hipStreamDestroy(stream));
  Check(hipFree(source));
}
