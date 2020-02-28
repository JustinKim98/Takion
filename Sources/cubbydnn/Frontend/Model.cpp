// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Frontend/Model.hpp>

namespace CubbyDNN
{
Unit Model::Constant(Shape shape, void* data)
{
    Unit unit(UnitType::Constant);
    unit.OutputTensorInfo = { shape, m_numberSystem };
    unit.Data = data;
    const auto id = m_unitVector.size();
    unit.ID = id;
    m_unitVector.emplace_back(unit);
    return unit;
}

Unit Model::Variable(Shape shape, Initializer initializer)
{
    Unit unit(UnitType::Variable);
    unit.OutputTensorInfo = { shape, m_numberSystem };
    // TODO : run Initializer here
    unit.Data = nullptr;
    const auto id = m_unitVector.size();
    m_unitVector.emplace_back(unit);
    return unit;
}

Unit Model::Add(const Unit& inputA, const Unit& inputB)
{
    const auto inputShapeA = inputA.OutputTensorInfo.GetShape();
    const auto inputShapeB = inputB.OutputTensorInfo.GetShape();

    assert(inputShapeA.Row == inputShapeB.Row);
    assert(inputShapeA.Col == inputShapeB.Col);
    assert(inputShapeA.Channel == inputShapeB.Channel);
    assert(inputShapeA.Batch == inputShapeB.Batch);

    const TensorInfo outputTensorInfo{ inputShapeA, m_numberSystem };
    Unit unit(UnitType::Add);
    unit.InputTensorInfoVector = { inputShapeA, inputShapeB };
    unit.OutputTensorInfo =
        outputTensorInfo;
    unit.InputIdVector = { inputA.ID, inputB.ID };
    unit.ID = m_unitVector.size();
    m_unitVector.emplace_back(unit);
    return unit;
}

Unit Model::Mul(const Unit& inputA, const Unit& inputB)
{
    const auto inputShapeA = inputA.OutputTensorInfo.GetShape();
    const auto inputShapeB = inputB.OutputTensorInfo.GetShape();

    assert(inputShapeA.Row == inputShapeB.Row);
    assert(inputShapeA.Col == inputShapeB.Col);
    assert(inputShapeA.Channel == inputShapeB.Channel);
    assert(inputShapeA.Batch == inputShapeB.Batch);

    const TensorInfo outputTensorInfo{ inputShapeA, m_numberSystem };
    Unit unit(UnitType::Add);
    unit.InputTensorInfoVector = { inputShapeA, inputShapeB };
    unit.OutputTensorInfo = outputTensorInfo;
    unit.InputIdVector = { inputA.ID, inputB.ID };
    unit.ID = m_unitVector.size();
    m_unitVector.emplace_back(unit);
    return unit;
}

Unit Model::Dense(const Unit& previous, size_t units, bool use_bias,
                  Initializer kernelInitializer,
                  Initializer biasInitializer)
{
    
}
} // namespace CubbyDNN
