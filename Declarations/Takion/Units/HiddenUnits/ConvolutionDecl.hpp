// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_CONVOLUTION_DECL_HPP
#define TAKION_COMPUTE_CONVOLUTION_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/FrontEnd/UnitMetaData.hpp>
#include <Takion/Units/TrainableUnit.hpp>

namespace Takion::Graph
{
template <typename T>
class Conv2DUnit : public ComputableUnit<T>, public TrainableUnit<T>
{
public:
    using ComputableUnit<T>::ForwardInputMap;
    using TrainableUnit<T>::TrainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;
    using ComputableUnit<T>::InternalTensorMap;
    using TrainableUnit<T>::m_optimizer;

    Conv2DUnit(const UnitId& unitId, const UnitId& sourceUnitId,
                  Tensor<T> forwardInput,
                  std::unordered_map<UnitId, Tensor<T>> backwardInputMap,
                  Tensor<T> forwardOutput, Tensor<T> backwardOutput,
                  std::unordered_map<std::string, Tensor<T>> internalTensorMap,
                  std::unordered_map<std::string, Tensor<T>> trainableTensorMap,
                  std::size_t dilation, std::size_t stride, std::size_t padding,
                  std::unique_ptr<Compute::Optimizer<T>> optimizer,
                  std::size_t batchSize);

    ~Conv2DUnit() = default;

    Conv2DUnit(const Conv2DUnit<T>& convolution2d) = delete;
    Conv2DUnit(Conv2DUnit<T>&& convolution2d) noexcept;
    Conv2DUnit& operator=(const Conv2DUnit<T>& convolution2D) = delete;
    Conv2DUnit& operator=(Conv2DUnit<T>&& convolutioin2d) noexcept;

    static Conv2DUnit<T> CreateUnit(
        const FrontEnd::UnitMetaData<T>& unitMetaData,
        std::unique_ptr<Compute::Optimizer<T>> optimizer);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

    void ChangeBatchSize(std::size_t batchSize) override;

private:
    static void m_checkShape(Shape input, Shape output, Shape filter,
                             Shape bias,
                             std::size_t dilationRow, std::size_t dilationCol,
                             std::size_t strideRow, std::size_t strideCol,
                             std::size_t padSizeRow, std::size_t padSizeCol,
                             std::string unitName);

    //! Assumes input Tensor is already padded
    void m_inputToInputMatrix(const Tensor<T>& input,
                              Tensor<T>& inputMatrix, Shape filterShape,
                              Shape outputShape, std::size_t dilation,
                              std::size_t rowStride, std::size_t colStride);

    void m_inputMatrixToInput(const Tensor<T>& inputMatrix, Tensor<T>& input,
                              Shape filterShape, Shape outputShape,
                              std::size_t dilation, std::size_t rowStride,
                              std::size_t colStride);

    //! Filter shape is in NCHW format where N is number of filters
    void m_filterToFilterMatrix(const Tensor<T>& filter,
                                Tensor<T>& filterMatrix);

    void m_filterMatrixToFilter(const Tensor<T>& filterMatrix,
                                Tensor<T>& filter);

    void m_outputToOutputMatrix(const Tensor<T>& output,
                                Tensor<T>& outputMatrix);

    void m_outputMatrixToOutput(const Tensor<T>& outputMatrix,
                                Tensor<T>& output);

    void m_biasToBiasMatrix(const Tensor<T>& bias, Tensor<T>& biasMatrix);

    void m_biasMatrixToBias(const Tensor<T>& biasMatrix, Tensor<T>& bias);

    void m_padForwardInput(const Tensor<T>& forwardInput,
                           Tensor<T>& paddedForwardInput);

    UnitId m_sourceUnitId;
};
}

#endif
