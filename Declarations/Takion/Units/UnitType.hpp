// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_UNITTYPE_HPP
#define TAKION_UNITTYPE_HPP

#include <Takion/Utils/SharedPtr.hpp>
#include <unordered_map>
#include <string>
#include <string_view>

namespace Takion
{
enum class UnitBaseType
{
    Fetcher,
    Constant,
    Hidden,
    Activation,
    Loss,
    Undefined,
};


class UnitType
{
public:
    UnitType() = default;
    UnitType(UnitBaseType type, std::string_view typeName);
    UnitType(UnitBaseType type, std::string_view name,
             SharedPtr<UnitType> baseUnit);
    ~UnitType() = default;

    UnitType(const UnitType& unitType) = default;
    UnitType(UnitType&& unitType) noexcept = default;
    UnitType& operator=(const UnitType& unitType) = default;
    UnitType& operator=(UnitType&& unitType) noexcept = default;

    bool operator==(const UnitType& unitType) const;
    bool operator!=(const UnitType& unitType) const;

    [[nodiscard]] const std::string& Name() const
    {
        return m_typeName;
    }

    [[nodiscard]] bool IsBaseOf(const UnitType& derivedUnit) const
    {
        return IsBaseOf(*this, derivedUnit);
    }

    [[nodiscard]] bool IsDerivedFrom(const UnitType& baseUnit) const
    {
        return IsBaseOf(baseUnit, *this);
    }

    static bool IsBaseOf(const UnitType& baseUnit, const UnitType& derivedUnit);

    SharedPtr<UnitType> BaseUnit;
    UnitBaseType BaseType = UnitBaseType::Undefined;

private:
    std::string m_typeName = "UnknownType";
};

struct UnitId
{
    UnitId() = default;

    UnitId(UnitType type, std::size_t id, std::string name)
        : Type(std::move(type)),
          Id(id),
          UnitName(std::move(name))
    {
    }

    bool operator==(const UnitId& unitId) const
    {
        return Type == unitId.Type && Id == unitId.Id &&
               UnitName == unitId.UnitName;
    }

    bool operator!=(const UnitId& unitId) const
    {
        return !(*this == unitId);
    }

    friend bool operator<(const UnitId& lhs, const UnitId& rhs)
    {
        return lhs.Id < rhs.Id;
    }

    friend bool operator<=(const UnitId& lhs, const UnitId& rhs)
    {
        return !(rhs < lhs);
    }

    friend bool operator>(const UnitId& lhs, const UnitId& rhs)
    {
        return rhs < lhs;
    }

    friend bool operator>=(const UnitId& lhs, const UnitId& rhs)
    {
        return !(lhs < rhs);
    }

    UnitType Type;
    std::size_t Id = 0;
    std::string UnitName = "Undefined";
};

//! UnitState
//! Wrapper class containing the state and ForwardStateCount
//! This represents the execution state of computable Unit
struct UnitState
{
    /// State number of current
    std::atomic<std::size_t> ForwardStateCount = 0;
    std::atomic<std::size_t> BackwardStateCount = 0;
};
} // namespace Takion

namespace std
{
template <>
struct hash<Takion::UnitId>
{
    std::size_t operator()(Takion::UnitId const& s) const noexcept
    {
        const std::size_t h1 = std::hash<std::string>{}(s.UnitName);
        const std::size_t h2 = std::hash<std::size_t>{}(s.Id);
        return h1 ^ (h2 << 1); // or use boost::hash_combine
    }
};
}; // namespace std
#endif
//! AddCpu template specialization for UnitId hash
