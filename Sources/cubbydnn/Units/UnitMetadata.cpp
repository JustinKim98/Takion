// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/UnitMetadata.hpp>

namespace CubbyDNN::Graph
{
UnitMetaData::UnitMetaData(
    UnitId unitId,
    std::unordered_map<std::string, Shape> internalVariableShapeMap,
    std::unordered_map<std::string, std::unique_ptr<Initializer>>
    initializerVector,
    std::vector<Shape> inputShapeVector, Shape outputShape,
    std::vector<UnitId> inputUnitIdVector,
    std::vector<UnitId> outputUnitIdVector, NumberSystem numericType,
    Compute::Device device, std::size_t padSize)
    : NumericType(numericType),
      PadSize(padSize),
      Device(std::move(device)),
      m_unitId(std::move(unitId)),
      m_internalVariableShapeMap(std::move(internalVariableShapeMap)),
      m_initializerVector(std::move(initializerVector)),
      m_inputShapeVector(std::move(inputShapeVector)),
      m_outputShape(std::move(outputShape)),
      m_inputUnitVector(std::move(inputUnitIdVector)),
      m_outputUnitVector(std::move(outputUnitIdVector))
{
}

UnitId UnitMetaData::Id() const
{
    return m_unitId;
}

std::vector<Shape> UnitMetaData::InputShapeVector() const
{
    return m_inputShapeVector;
}

Shape UnitMetaData::OutputShape() const
{
    return m_outputShape;
}

std::vector<UnitId> UnitMetaData::InputUnitVector() const
{
    return m_inputUnitVector;
}

std::vector<UnitId> UnitMetaData::OutputUnitVector() const
{
    return m_outputUnitVector;
}

const std::unique_ptr<Initializer>& UnitMetaData::
GetInitializer(const std::string& name) const
{
    return m_initializerVector.at(name);
}

Shape UnitMetaData::GetShape(const std::string& name) const
{
    return m_internalVariableShapeMap.at(name);
}
}
