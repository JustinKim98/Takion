// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_UNITMETADATA_HPP
#define CUBBYDNN_UNITMETADATA_HPP

#include <cubbydnn/Units/UnitType.hpp>
#include <cubbydnn/Utils/Shape.hpp>
#include <cubbydnn/Utils/Declarations.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>
#include <vector>
#include <memory>
#include <unordered_map>

namespace CubbyDNN::Graph
{
class ParameterPack
{
public:
    ParameterPack() = default;
    ~ParameterPack() = default;

    ParameterPack(const ParameterPack& parameterPack) = default;
    ParameterPack(ParameterPack&& parameterPack) noexcept = default;
    ParameterPack& operator=(const ParameterPack& parameterPack) = default;
    ParameterPack& operator=(ParameterPack&& parameterPack) noexcept = default;

    ParameterPack(std::unordered_map<std::string, int> integerParams,
                  std::unordered_map<std::string, float> floatingPointParams,
                  std::unordered_map<std::string, std::string> stringParams);

    [[nodiscard]] int GetIntegerParam(const std::string& name) const;

    [[nodiscard]] float GetFloatingPointParam(const std::string& name) const;

    [[nodiscard]] std::string GetStringParam(const std::string& name) const;

private:
    std::unordered_map<std::string, int> m_integerParameters;
    std::unordered_map<std::string, float> m_floatingPointParameters;
    std::unordered_map<std::string, std::string> m_stringParameters;
};

class UnitMetaData
{
public:
    UnitMetaData(
        UnitId unitId,
        std::unordered_map<std::string, Shape> internalVariableShapeMap,
        std::unordered_map<std::string, std::unique_ptr<Initializer>>
        initializerMap,
        std::vector<Shape> inputShapeVector, Shape outputShape,
        std::vector<UnitId> inputUnitIdVector,
        NumberSystem numericType, Compute::Device device);


    UnitMetaData(
        UnitId unitId,
        std::unordered_map<std::string, Shape> internalVariableShapeMap,
        std::unordered_map<std::string, std::unique_ptr<Initializer>>
        initializerMap,
        std::vector<Shape> inputShapeVector, Shape outputShape,
        std::vector<UnitId> inputUnitIdVector, ParameterPack params,
        NumberSystem numericType,
        Compute::Device device);

    ~UnitMetaData() = default;

    UnitMetaData(const UnitMetaData& unitMetaData) = delete;

    UnitMetaData(UnitMetaData&& unitMetaData) noexcept;
    UnitMetaData& operator=(const UnitMetaData& unitMetaData) = delete;
    UnitMetaData& operator=(UnitMetaData&& unitMetaData) noexcept;

    void AppendOutputUnitId(UnitId unitId);

    void SetOutputUnitIdVector(std::vector<UnitId> unitIdVector);

    [[nodiscard]] UnitId Id() const;

    [[nodiscard]] std::vector<Shape> InputShapeVector() const;

    [[nodiscard]] Shape OutputShape() const;

    [[nodiscard]] std::vector<UnitId> InputUnitVector() const;

    [[nodiscard]] std::vector<UnitId> OutputUnitVector() const;

    [[nodiscard]] const std::unique_ptr<Initializer>& GetInitializer(
        const std::string& name) const;

    [[nodiscard]] Shape GetShape(const std::string& name) const;

    ParameterPack Parameters = ParameterPack();
    NumberSystem NumericType;
    Compute::Device Device;

private:
    UnitId m_unitId;
    std::unordered_map<std::string, Shape> m_internalVariableShapeMap;
    std::unordered_map<std::string, std::unique_ptr<Initializer>>
    m_initializerMap;

    std::vector<Shape> m_inputShapeVector;
    Shape m_outputShape;

    std::vector<UnitId> m_inputUnitIdVector;
    std::vector<UnitId> m_outputUnitIdVector;
};
}
#endif