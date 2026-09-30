#include "src/models/qwen38_flash_next/prompt_lookup.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using gufo::models::qwen38_flash_next::PromptLookup;

int main() {
  PromptLookup lookup;
  const std::array<std::int32_t, 2> suffix{12, 13};
  std::vector<std::int32_t> history{10, 11, 12, 13, 20, 21, 22,
                                    23, 24, 25, 99, 10, 11};
  const auto expect = [&](std::span<const std::int32_t> actual,
                          std::initializer_list<std::int32_t> wanted) {
    assert(std::ranges::equal(actual, wanted));
  };
  expect(lookup.Propose(history, suffix, 6), {20, 21, 22, 23, 24, 25});
  expect(lookup.Propose(history, suffix, 2), {20, 21});
  assert(lookup.Propose(history, suffix, 1).empty());
  assert(
      lookup.Propose(history, std::array<std::int32_t, 2>{12, 14}, 6).empty());

  // The newest complete occurrence replaces an older continuation. An
  // incomplete occurrence at the frontier must not shadow it.
  history.insert(history.end(), {12, 13, 30, 31, 32, 33, 34, 35, 99, 10, 11});
  expect(lookup.Propose(history, suffix, 6), {30, 31, 32, 33, 34, 35});
  PromptLookup restored;
  expect(restored.Propose(history, suffix, 6), {30, 31, 32, 33, 34, 35});
  lookup.Reset();
  expect(lookup.Propose(history, suffix, 99), {30, 31, 32, 33, 34, 35});
  lookup.Reset();
  assert(lookup.Propose(std::array<std::int32_t, 1>{10}, suffix, 6).empty());

  // Force table collisions with more distinct keys than slots. Every hit
  // still points to the exact requested four tokens, never a hash alias.
  history.resize(100000);
  std::iota(history.begin(), history.end(), 0);
  std::size_t hits = 0, misses = 0;
  for (std::int32_t end = 4; end < 99994; ++end) {
    const std::array key{end - 4, end - 3, end - 2, end - 1};
    const auto proposal = lookup.Propose(history, key, 6);
    if (proposal.empty()) {
      ++misses;
    } else {
      ++hits;
      for (std::size_t i = 0; i < proposal.size(); ++i)
        assert(proposal[i] == end + static_cast<std::int32_t>(i));
    }
  }
  assert(hits > 0 && misses > 0);
  std::cout
      << "prompt lookup: bounds, append, reset, restore and collisions pass\n";
}
