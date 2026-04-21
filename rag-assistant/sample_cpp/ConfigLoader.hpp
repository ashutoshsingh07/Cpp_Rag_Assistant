/**
 * ConfigLoader.hpp - Sample C++ file for testing the RAG assistant
 * This is a realistic example of the kind of code the assistant will analyze.
 */
#pragma once
#include <string>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <stdexcept>

namespace config {

/**
 * ConfigLoader handles loading and parsing XML/XSD-based configuration files.
 * Supports multiple product variants via variant-specific override files.
 *
 * Usage:
 *   auto loader = ConfigLoader("/etc/app/config.xml");
 *   loader.loadVariant("philippines");
 *   auto value = loader.get<int>("game.rtp");
 */
class ConfigLoader {
public:
    using ConfigMap = std::unordered_map<std::string, std::string>;

    explicit ConfigLoader(const std::filesystem::path& base_config_path)
        : base_path_(base_config_path) {
        if (!std::filesystem::exists(base_config_path)) {
            throw std::runtime_error("Config file not found: " + base_config_path.string());
        }
    }

    /// Load base configuration from XML file
    bool load() {
        return parseXML(base_path_, base_config_);
    }

    /// Load variant-specific overrides on top of base config
    bool loadVariant(const std::string& variant_name) {
        auto variant_path = base_path_.parent_path() / (variant_name + ".xml");
        if (!std::filesystem::exists(variant_path)) {
            return false;  // Variant not found — not an error, use base config
        }

        ConfigMap variant_config;
        if (!parseXML(variant_path, variant_config)) {
            return false;
        }

        // Merge: variant overrides base
        for (auto& [key, value] : variant_config) {
            base_config_[key] = value;
        }
        return true;
    }

    /// Get a typed value from config. Returns std::nullopt if key not found.
    template <typename T>
    std::optional<T> get(const std::string& key) const;

    /// Get value with default fallback
    template <typename T>
    T getOr(const std::string& key, const T& default_value) const {
        auto val = get<T>(key);
        return val.has_value() ? *val : default_value;
    }

    size_t size() const { return base_config_.size(); }

private:
    std::filesystem::path base_path_;
    ConfigMap base_config_;

    bool parseXML(const std::filesystem::path& path, ConfigMap& out);
};

// Template specializations
template <>
std::optional<int> ConfigLoader::get<int>(const std::string& key) const {
    auto it = base_config_.find(key);
    if (it == base_config_.end()) return std::nullopt;
    try { return std::stoi(it->second); }
    catch (...) { return std::nullopt; }
}

template <>
std::optional<std::string> ConfigLoader::get<std::string>(const std::string& key) const {
    auto it = base_config_.find(key);
    if (it == base_config_.end()) return std::nullopt;
    return it->second;
}

} // namespace config
