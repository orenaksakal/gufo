#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace gufo::models::qwen38_flash_next {

// A bounded, session-private index of committed text. Proposals are only hints:
// the target verifier must check every token. Hash collisions can lose a match
// but cannot invent one. No weights, KV entries or logits are approximated.
class PromptLookup {
public:
  static constexpr std::size_t kMatch = 4;
  static constexpr std::size_t kTail = 6;

  void Reset() noexcept {
    positions_.clear();
    indexed_ = kMatch;
  }

  // history is append-only between Reset calls. suffix contains the target
  // anchor and the first MTP proposal; neither is inserted into the index.
  // Rebuilding from committed history after restore gives the same proposals.
  [[nodiscard]] std::span<const std::int32_t> Propose(
      std::span<const std::int32_t> history,
      std::span<const std::int32_t> suffix, std::size_t budget) {
    if (budget < 2 || history.size() < kMatch + kTail ||
        history.size() > std::numeric_limits<std::uint32_t>::max())
      return {};
    if (positions_.empty())
      positions_.assign(kSlots, kMissing);
    // Keep only positions with a complete continuation, so a recent match
    // without enough successors cannot hide an older, useful copy.
    for (; indexed_ + kTail <= history.size(); ++indexed_) {
      const auto key = history.subspan(indexed_ - kMatch, kMatch);
      positions_[Slot(key)] = static_cast<std::uint32_t>(indexed_);
    }
    std::array<std::int32_t, kMatch> key;
    const auto extra = std::min(suffix.size(), kMatch);
    std::copy(history.end() - (kMatch - extra), history.end(), key.begin());
    std::copy(suffix.end() - extra, suffix.end(), key.end() - extra);
    const auto position = positions_[Slot(key)];
    if (position == kMissing || position > history.size() ||
        !std::equal(key.begin(), key.end(),
                    history.begin() + position - kMatch))
      return {};
    return history.subspan(
        position, std::min({budget, kTail, history.size() - position}));
  }

private:
  static constexpr std::size_t kSlots = 1U << 16;
  static constexpr std::uint32_t kMissing =
      std::numeric_limits<std::uint32_t>::max();
  static std::size_t Slot(std::span<const std::int32_t> key) noexcept {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const auto token : key)
      hash = (hash ^ static_cast<std::uint32_t>(token)) * 0x100000001b3ULL;
    return (hash ^ (hash >> 32)) & (kSlots - 1);
  }
  std::vector<std::uint32_t> positions_;
  std::size_t indexed_{kMatch};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_
