// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_CONVOLUTION_HPP
#define TAKION_COMPUTE_CONVOLUTION_HPP
#include <Takion/Units/HiddenUnits/ConvolutionDecl.hpp>

namespace Takion::Graph
{
template <typename T>
Conv2DUnit<T>::Conv2DUnit(const UnitId& unitId,
                          const UnitId& sourceUnitId,
                          Tensor<T> forwardInput,
                          std::unordered_map<UnitId, Tensor<T>>
                          backwardInputMap, Tensor<T> forwardOutput,
                          Tensor<T> backwardOutput,
                          std::unordered_map<std::string, Tensor<T>>
                          internalTensorMap,
                          std::unordered_map<std::string, Tensor<T>>
                          trainableTensorMap,
                          std::size_t dilation, std::size_t stride,
                          std::size_t padding,
                          std::unique_ptr<Compute::Optimizer<T>>
                          optimizer, std::size_t batchSize)
    : ComputableUnit<T>(unitId, { { sourceUnitId, std::move(forwardInput) } },
                        std::move(backwardInputMap), forwardOutput,
                        { { sourceUnitId, std::move(backwardOutput) } },
                        std::move(internalTensorMap), batchSize),
      TrainableUnit<T>(std::move(trainableTensorMap), std::move(optimizer)),
      m_sourceUnitId(sourceUnitId)
{
    Tensor<T>& filter = TrainableTensorMap["filter"];
    Tensor<T>& filterMatrix = TrainableTensorMap["filterMatrix"];
    m_filterToFilterMatrix(filter, filterMatrix);

    Tensor<T>& forwardInputTensor = ForwardInputMap[m_sourceUnitId];
    Tensor<T>& paddedForwardInput = InternalTensorMap["paddedForwardInput"];
    m_padForwardInput(forwardInputTensor, paddedForwardInput);
}

template <typename T>
Conv2DUnit<T> Conv2DUnit<T>::CreateUnit(
    const FrontEnd::UnitMetaData<T>& unitMetaData,
    std::unique_ptr<Compute::Optimizer<T>> optimizer)
{
    const auto unitId = unitMetaData.Id();
    auto sourceUnitId = unitMetaData.GetInputUnitId("input");
    const auto batchSize = unitMetaData.BatchSize();
    const auto filterShape = unitMetaData.InternalVariableShape("filter");
    const auto biasShape = unitMetaData.InternalVariableShape("bias");
    const auto inputShape = unitMetaData.GetInputShape("input");
    const auto forwardOutputShape = unitMetaData.GetOutputShape();
    const std::size_t dilation =
        unitMetaData.Params.GetIntegerParam("dilation");
    const auto stride = unitMetaData.Params.GetIntegerParam("stride");
    const std::size_t padSizeX = unitMetaData
                                 .Params.GetIntegerParam("padSizeX");
    const std::size_t padSizeY =
        unitMetaData.Params.GetIntegerParam("padSizeY");

    const auto& filterInitializer = unitMetaData.GetInitializer("filter");
    const auto& biasInitializer = unitMetaData.GetInitializer("bias");

    const auto device = unitMetaData.Device;

    const auto filterNumCol = filterShape[3];
    const auto filterNumRow = filterShape[2];
    const auto inputNumChannel = filterShape[1];
    const auto outputNumChannel = filterShape[0];

    const auto outputMapSize = forwardOutputShape[1] * forwardOutputShape[2];

    Shape filterMatrixShape{
        filterNumCol * filterNumRow * inputNumChannel,
        outputNumChannel };

    Shape inputForwardMatrixShape{ outputMapSize,
                                   filterNumCol * filterNumRow *
                                   inputNumChannel };

    Shape backwardOutputMatrixShape = inputForwardMatrixShape;
    Shape forwardOutputMatrixShape{ outputMapSize, outputNumChannel };

    Shape biasMatrixShape = forwardOutputMatrixShape;

    Shape paddedInputShape = inputShape;
    paddedInputShape.SetNumCols(inputShape.NumCol() + 2 * padSizeX);
    paddedInputShape.SetNumRows(inputShape.NumRow() + 2 * padSizeY);

    Tensor<T> forwardInput(inputShape, batchSize, device);

    Tensor<T> paddedForwardInput(paddedInputShape, batchSize, device);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;

    for (const auto& outputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor<T> tensor(forwardOutputShape, batchSize, device);
        backwardInputMap[outputUnitId] = tensor;
    }

    Tensor<T> forwardOutput(forwardOutputShape, batchSize, device);
    Tensor<T> backwardOutput(inputShape, batchSize, device);

    Tensor<T> forwardInputMatrix(inputForwardMatrixShape, batchSize, device);
    Tensor<T> backwardOutputMatrix(backwardOutputMatrixShape, batchSize,
                                   device);

    Tensor<T> forwardOutputMatrix(forwardOutputMatrixShape, batchSize);

    Tensor<T> filter(filterShape, device);

    Tensor<T> filterMatrix(filterMatrixShape, device);
    Tensor<T> filterUpdateMatrix(filterMatrixShape, batchSize, device);
    Tensor<T> filterUpdateMatrixMean(filterMatrixShape, device);
    Tensor<T> transposedFilterMatrix(filterMatrixShape.GetTranspose(),
                                     batchSize,
                                     device);

    Tensor<T> bias(biasShape, device);
    Tensor<T> biasMatrix(biasMatrixShape, device);
    Tensor<T> biasUpdateMatrixMean(biasMatrixShape, device);

    Tensor<T> delta(forwardOutputShape, batchSize, device);
    Tensor<T> deltaMatrix(forwardOutputMatrixShape, batchSize, device);

    filterInitializer->Initialize(filterMatrix);
    biasInitializer->Initialize(biasMatrix);

    std::unordered_map<std::string, Tensor<T>> trainableTensorMap = {
        { "filter", filter },
        { "bias", bias },
        { "filterMatrix", filterMatrix },
        { "biasMatrix", biasMatrix }
    };

    std::unordered_map<std::string, Tensor<T>> internalTensorMap = {
        { "paddedForwardInput", paddedForwardInput },
        { "forwardInputMatrix", forwardInputMatrix },
        { "forwardOutputMatrix", forwardOutputMatrix },
        { "backwardOutputMatrix", backwardOutputMatrix },
        { "delta", delta },
        { "deltaMatrix", deltaMatrix },
        { "transposedFilterMatrix", transposedFilterMatrix },

    };

    auto conv2DUnit = Conv2DUnit<T>(
        unitId, sourceUnitId, forwardInput, backwardInputMap,
        forwardOutput, backwardOutput, internalTensorMap,
        trainableTensorMap, dilation, stride, padSizeX, optimizer, batchSize);

    return conv2DUnit;
}

template <typename T>
void Conv2DUnit<T>::Forward()
{
    const Tensor<T>& inputForward = ForwardInputMap[m_sourceUnitId];
    Tensor<T>& inputForwardMatrix = InternalTensorMap["forwardInputMatrix"];
    m_inputToInputMatrix(inputForward, inputForwardMatrix);
    Tensor<T>& filterMatrix = TrainableTensorMap["filterMatrix"];
    const Tensor<T>& biasMatrix = TrainableTensorMap["biasMatrix"];
    Tensor<T>& outputForwardMatrix = InternalTensorMap["forwardOutputMatrix"];

    Compute::Multiply(filterMatrix, inputForwardMatrix, outputForwardMatrix);
    Compute::Add(outputForwardMatrix, biasMatrix, outputForwardMatrix);
}

template <typename T>
void Conv2DUnit<T>::AsyncForward(std::promise<bool> promise)
{
    const Tensor<T>& inputForward = ForwardInputMap[m_sourceUnitId];
    Tensor<T>& inputForwardMatrix = InternalTensorMap["forwardInputMatrix"];
    m_inputToInputMatrix(inputForward, inputForwardMatrix);
    Tensor<T>& filterMatrix = TrainableTensorMap["filterMatrix"];
    const Tensor<T>& biasMatrix = TrainableTensorMap["biasMatrix"];
    Tensor<T>& outputForwardMatrix = InternalTensorMap["forwardOutputMatrix"];

    Compute::Multiply(filterMatrix, inputForwardMatrix, outputForwardMatrix);
    Compute::Add(outputForwardMatrix, biasMatrix, outputForwardMatrix);
    promise.set_value(true);
}

template <typename T>
void Conv2DUnit<T>::Backward()
{
    Tensor<T>& delta = InternalTensorMap["delta"];
    Tensor<T>& transposedFilterMatrix =
        TrainableTensorMap["transposedFilterMatrix"];
    Tensor<T>& filterMatrix = TrainableTensorMap["filterMatrix"];
    Tensor<T>& biasMatrix = TrainableTensorMap["biasMatrix"];
    const Tensor<T>& forwardInputMatrix =
        InternalTensorMap["forwardInputMatrix"];
    Tensor<T>& backwardOutputMatrix = InternalTensorMap["backwardOutputMatrix"];
    Tensor<T>& backwardOutput = BackwardOutputMap[m_sourceUnitId];
    Tensor<T>& filterUpdateMatrix = InternalTensorMap["filterUpdateMatrix"];
    Tensor<T>& filterUpdateMatrixMean =
        InternalTensorMap["filterUpdateMatrixMean"];
    Tensor<T>& biasUpdateMatrixMean = InternalTensorMap["biasUpdateMatrixMean"];

    const Compute::Zeros<T> zeroInitializer;
    zeroInitializer.Initialize(delta);

    for (auto& [unitId, gradient] : BackwardInputMap)
    {
        Compute::Add(gradient, delta);
    }
    Compute::ScalarDiv(delta, static_cast<T>(BackwardInputMap.size()));
    Compute::Transpose(filterMatrix, transposedFilterMatrix);
    Compute::Multiply(transposedFilterMatrix, delta, backwardOutputMatrix);

    m_inputMatrixToInput(backwardOutputMatrix, backwardOutput);

    Compute::Multiply(filterMatrix, forwardInputMatrix, filterUpdateMatrix);
    Compute::Shrink(filterUpdateMatrix, filterUpdateMatrixMean);
    Compute::Shrink(delta, biasUpdateMatrixMean);

    m_optimizer->Optimize(filterMatrix, filterUpdateMatrixMean);
    m_optimizer->Optimize(biasMatrix, biasUpdateMatrixMean);
}


template <typename T>
void Conv2DUnit<T>::m_checkShape(Shape input, Shape output, Shape filter,
                                 Shape bias, std::size_t dilationRow,
                                 std::size_t dilationCol,
                                 std::size_t strideRow,
                                 std::size_t strideCol,
                                 std::size_t padSizeRow,
                                 std::size_t padSizeCol,
                                 std::string unitName)
{
    if (input.Dim() != 3)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - input dimension should be 3 - (numChannels, numRow, numColumn)";
        throw std::invalid_argument(errorMessage);
    }
    if (output.Dim() != 3)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - output dimension should be 3 - (numChannels, numRow, numColumn)";
        throw std::invalid_argument(errorMessage);
    }
    if (filter.Dim() != 4)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - filter dimension should be 4 - (numOutputChannels, numInputChannels, numRow, numColumn)";
        throw std::invalid_argument(errorMessage);
    }
    if (bias.Dim() != 1)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - bias dimension should be 1 - (numOutputChannels)";
        throw std::invalid_argument(errorMessage);
    }

    const auto filterNumCol =
        filter.NumCol() + (filter.NumCol() - 1) * dilationCol;
    const auto filterNumRow =
        filter.NumRow() + (filter.NumRow() - 1) * dilationRow;

    if (input.NumRow() + 2 * padSizeRow > filterNumRow)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - filter size with dilation cannot be larger than input size. Given filter row size with dilation : "
            +
            std::to_string(filterNumRow) + " Given padded input row size : " +
            std::to_string(input.NumRow() + 2 * padSizeRow);
        throw std::invalid_argument(errorMessage);
    }
    if (input.NumCol() + 2 * padSizeCol > filterNumCol)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - filter size with dilation cannot be larger than input size. "
            "Given filter column size with dilation : " +
            std::to_string(filterNumCol) + " Given padded input column size : "
            +
            std::to_string(input.NumCol() + 2 * padSizeCol);
        throw std::invalid_argument(errorMessage);
    }

    const auto expectedOutputNumRow =
        (input.NumRow() - filterNumRow + 2 * padSizeRow) / strideRow;
    const auto expectedOutputNumCol =
        (input.NumCol() - filterNumCol + 2 * padSizeCol) / strideCol;

    const auto expectedOutputChannelSize = filter[0];

    if (bias[0] != expectedOutputChannelSize)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - bias should be 1 dimensional tensor with size : " +
            std::to_string(expectedOutputChannelSize) +
            " While given : " + std::to_string(bias[0]);
        throw std::invalid_argument(errorMessage);
    }
    if (output[0] != expectedOutputChannelSize)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - output should be 3 dimensional tensor with channel size : " +
            std::to_string(expectedOutputChannelSize) +
            " While given : " + std::to_string(output[0]);
        throw std::invalid_argument(errorMessage);
    }
    if (output[1] != expectedOutputNumRow)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - output should be 3 dimensional tensor with row size : " +
            std::to_string(expectedOutputNumRow) +
            " While given : " + std::to_string(output[1]);
        throw std::invalid_argument(errorMessage);
    }
    if (output[2] != expectedOutputNumCol)
    {
        const std::string errorMessage =
            std::string("Conv2D ") + unitName +
            " - output should be 3 dimensional tensor with column size : " +
            std::to_string(expectedOutputNumCol) +
            " While given : " + std::to_string(output[2]);
        throw std::invalid_argument(errorMessage);
    }
}


template <typename T>
void Conv2DUnit<T>::m_inputToInputMatrix(
    const Tensor<T>& input, Tensor<T>& inputMatrix, Shape filterShape,
    Shape outputShape, std::size_t dilation, std::size_t rowStride,
    std::size_t colStride)
{
    const auto batchSize = input.BatchSize;
    const auto totalInputSize = input.TensorShape.Size();
    const auto matrixSize = inputMatrix.TensorShape.Size();
    const auto matrixNumCol = inputMatrix.TensorShape.NumCol();
    const auto matrixMapSize =
        inputMatrix.TensorShape.NumCol() * inputMatrix.TensorShape.NumRow();
    const auto filterMapSize = filterShape.NumCol() * filterShape.NumRow();

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto inputTensorBatchOffset = totalInputSize * batchSize;
        const auto inputMatrixBatchOffset = inputMatrix * batchSize;

        for (std::size_t inputIdxWithBatch = inputMatrixBatchOffset;
             inputIdxWithBatch < inputMatrixBatchOffset + matrixSize;
             ++inputIdxWithBatch)
        {
            const auto inputIndex = inputIdxWithBatch % matrixSize;
            const auto matrixRowIdx = inputIndex / matrixNumCol;
            const auto matrixColIdx = inputIndex % matrixNumCol;
            const auto mapPositionRow = matrixRowIdx / outputShape.NumCol();
            const auto mapPositionCol = matrixRowIdx % outputShape.NumCol();
            const auto channelIdx = matrixColIdx / filterMapSize;
            const auto mapInternalIdx = inputIndex % filterMapSize;
            auto mapRow = mapInternalIdx / filterShape.NumCol();
            auto mapCol = mapInternalIdx % filterShape.NumCol();

            mapRow = mapRow + mapRow * dilation;
            mapCol = mapCol + mapCol * dilation;

            const auto pos =
                inputTensorBatchOffset + channelIdx * matrixMapSize +
                matrixNumCol * (mapPositionRow * rowStride + mapRow) +
                mapPositionCol * colStride + mapCol;

            inputMatrix.At(inputIdxWithBatch) = input.At(pos);
        }
    }
}

template <typename T>
void Conv2DUnit<T>::m_inputMatrixToInput(const Tensor<T>& inputMatrix,
                                         Tensor<T>& input,
                                         Shape filterShape,
                                         Shape outputShape,
                                         std::size_t dilation,
                                         std::size_t rowStride,
                                         std::size_t colStride)
{
    const auto batchSize = input.BatchSize;
    const auto totalInputSize = input.TensorShape.Size();
    const auto matrixSize = inputMatrix.TensorShape.Size();
    const auto matrixNumCol = inputMatrix.TensorShape.NumCol();
    const auto matrixMapSize =
        inputMatrix.TensorShape.NumCol() * inputMatrix.TensorShape.NumRow();
    const auto filterMapSize = filterShape.NumCol() * filterShape.NumRow();

    Compute::Zeros<T> zeroInitializer;
    zeroInitializer.Initialize(input);

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto inputTensorBatchOffset = totalInputSize * batchSize;
        const auto inputMatrixBatchOffset = inputMatrix * batchSize;

        for (std::size_t inputIdxWithBatch = inputMatrixBatchOffset;
             inputIdxWithBatch < inputMatrixBatchOffset + matrixSize; ++
             inputIdxWithBatch)
        {
            const auto inputIndex = inputIdxWithBatch % matrixSize;
            const auto matrixRowIdx = inputIndex / matrixNumCol;
            const auto matrixColIdx = inputIndex % matrixNumCol;
            const auto mapPositionRow = matrixColIdx / outputShape.NumCol();
            const auto mapPositionCol = matrixColIdx % outputShape.NumCol();
            const auto channelIdx = matrixRowIdx / filterMapSize;
            const auto mapInternalIdx = inputIndex % filterMapSize;
            auto mapRow = mapInternalIdx / filterShape.NumCol();
            auto mapCol = mapInternalIdx % filterShape.NumCol();

            mapRow = mapRow + mapRow * dilation;
            mapCol = mapCol + mapCol * dilation;

            const auto pos =
                inputTensorBatchOffset + channelIdx * matrixMapSize +
                matrixNumCol * (mapPositionRow * rowStride + mapRow) +
                mapPositionCol * colStride + mapCol;

            input.At(pos) += inputMatrix.At(inputIdxWithBatch);
        }
    }
}

template <typename T>
void Conv2DUnit<T>::m_outputToOutputMatrix(const Tensor<T>& output,
                                           Tensor<T>& outputMatrix)
{
    const auto batchSize = output.BatchSize;
    const auto outputSize = output.TensorShape.Size();
    const auto outputMatrixSize = outputMatrix.TensorShape.Size();
    const auto outputMatrixShape = outputMatrix.TensorShape;
    const auto outputShape = output.TensorShape;
    const auto outputMapSize = outputShape.NumRow() * outputShape.NumCol();

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto outputMatrixBatchOffset = outputMatrixSize * batchIdx;
        const auto outputBatchOffset = outputSize * batchIdx;

        for (std::size_t matrixIdx = outputMatrixBatchOffset;
             matrixIdx < outputMatrixBatchOffset + outputMatrixSize; ++matrixIdx
        )
        {
            const auto index = matrixIdx % outputShape.Size();
            const auto matrixRow = index / outputMatrixShape.NumCol();
            const auto matrixCol = index % outputMatrixShape.NumCol();

            const auto channelIdx = matrixRow;
            const auto rowIdx = matrixCol / outputShape.NumCol();
            const auto colIdx = matrixCol % outputShape.NumCol();

            const auto pos = outputBatchOffset + channelIdx * outputMapSize +
                             rowIdx * outputShape.NumCol() + colIdx;

            output.At(pos) = outputMatrix.At(matrixIdx);
        }
    }
}

template <typename T>
void Conv2DUnit<T>::m_outputMatrixToOutput(const Tensor<T>& outputMatrix,
                                           Tensor<T>& output)
{
    const auto batchSize = output.BatchSize;
    const auto outputSize = output.TensorShape.Size();
    const auto outputMatrixSize = outputMatrix.TensorShape.Size();
    const auto outputMatrixShape = outputMatrix.TensorShape;
    const auto outputShape = output.TensorShape;
    const auto outputMapSize = outputShape.NumRow() * outputShape.NumCol();

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto outputMatrixBatchOffset = outputMatrixSize * batchIdx;
        const auto outputBatchOffset = outputSize * batchIdx;

        for (std::size_t matrixIdx = outputMatrixBatchOffset;
             matrixIdx < outputMatrixBatchOffset + outputMatrixSize;
             ++matrixIdx)
        {
            const auto index = matrixIdx % outputShape.Size();
            const auto outputMatrixRow = index / outputMatrixShape.NumCol();
            const auto outputMatrixCol = index % outputMatrixShape.NumCol();

            const auto channelIdx = outputMatrixRow;
            const auto rowIdx = outputMatrixCol / outputShape.NumCol();
            const auto colIdx = outputMatrixCol % outputShape.NumCol();

            const auto pos = outputBatchOffset + channelIdx * outputMapSize +
                             rowIdx * outputShape.NumCol() + colIdx;

            outputMatrix.At(matrixIdx) = output.At(pos);
        }
    }
}


template <typename T>
void Conv2DUnit<T>::m_filterToFilterMatrix(const Tensor<T>& filter,
                                           Tensor<T>& filterMatrix)
{
    const auto batchSize = filter.BatchSize;
    const auto filterMatrixShape = filterMatrix.TensorShape;
    const auto filterShape = filter.TensorShape;
    const auto numChannels = filterShape[1];
    const auto filterMatrixSize = filterMatrixShape.Size();
    const auto filterTotalSize = filter.TensorShape.Size();
    const auto filterMapSize = filterShape.NumCol() * filterShape.NumRow();
    const auto filterSize = filterMapSize * numChannels;

    for (std::size_t matrixIdx = 0;
         matrixIdx < filterTotalSize; ++matrixIdx)
    {
        const auto matrixRow = matrixIdx / filterMatrixShape.NumCol();
        const auto matrixCol = matrixIdx % filterMatrixShape.NumRow();

        const auto filterIdx = matrixCol;
        const auto filterChannelIdx = matrixRow / filterMapSize;

        const auto filterInternalMapIdx = matrixRow % filterMapSize;

        const auto filterCol = filterInternalMapIdx % filterShape.NumCol();
        const auto filterRow = filterInternalMapIdx / filterShape.NumRow();

        const auto filterPos = filterIdx * filterSize +
                               filterChannelIdx * filterMapSize +
                               filterShape.NumCol() * filterRow + filterCol;

        filterMatrix.At(matrixIdx) = filter.At(filterPos);
    }
}

template <typename T>
void Conv2DUnit<T>::m_filterMatrixToFilter(const Tensor<T>& filterMatrix,
                                           Tensor<T>& filter)
{
    const auto batchSize = filter.BatchSize;
    const auto filterMatrixShape = filterMatrix.TensorShape;
    const auto filterShape = filter.TensorShape;
    const auto numChannels = filterShape[1];
    const auto filterMatrixSize = filterMatrixShape.Size();
    const auto filterTotalSize = filter.TensorShape.Size();
    const auto filterMapSize = filterShape.NumCol() * filterShape.NumRow();
    const auto filterSize = filterMapSize * numChannels;

    for (std::size_t matrixIdx = 0; matrixIdx < filterTotalSize; ++matrixIdx)
    {
        const auto matrixRow = matrixIdx / filterMatrixShape.NumCol();
        const auto matrixCol = matrixIdx % filterMatrixShape.NumRow();

        const auto filterIdx = matrixCol;
        const auto filterChannelIdx = matrixRow / filterMapSize;

        const auto filterInternalMapIdx = matrixRow % filterMapSize;

        const auto filterCol = filterInternalMapIdx % filterShape.NumCol();
        const auto filterRow = filterInternalMapIdx / filterShape.NumRow();

        const auto filterPos = filterIdx * filterSize +
                               filterChannelIdx * filterMapSize +
                               filterShape.NumCol() * filterRow + filterCol;

        filter.At(filterPos) = filterMatrix.At(matrixIdx);
    }
}

template <typename T>
void Conv2DUnit<T>::m_biasToBiasMatrix(const Tensor<T>& bias,
                                       Tensor<T>& biasMatrix)
{
    const auto batchSize = biasMatrix.BatchSize;
    const auto biasMatrixShape = biasMatrix.TensorShape;
    const auto mapSize = biasMatrixShape[1] * biasMatrixShape[2];
    const auto biasMatrixSize = biasMatrixShape.Size();
    const auto biasMatrixMapSize =
        biasMatrixShape.NumCol() * biasMatrixShape.NumRow();

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto biasMatrixBatchOffset = batchIdx * biasMatrixSize;
        for (std::size_t biasMatrixIndex = biasMatrixBatchOffset;
             biasMatrixIndex < biasMatrixBatchOffset + biasMatrixSize; ++
             biasMatrixIndex)
        {
            const auto matrixIdx = biasMatrixBatchOffset % biasMatrixSize;
            const auto channelIdx = matrixIdx % mapSize;
            biasMatrix.At(biasMatrixIndex) = bias.At(channelIdx);
        }
    }
}

template <typename T>
void Conv2DUnit<T>::m_biasMatrixToBias(const Tensor<T>& biasMatrix,
                                       Tensor<T>& bias)
{
    const auto batchSize = biasMatrix.BatchSize;
    const auto biasMatrixShape = biasMatrix.TensorShape;
    const auto mapSize = biasMatrixShape[1] * biasMatrixShape[2];
    const auto biasMatrixSize = biasMatrixShape.Size();
    const auto biasMatrixMapSize =
        biasMatrixShape.NumCol() * biasMatrixShape.NumRow();

    for (long batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        const auto biasMatrixBatchOffset = batchIdx * biasMatrixSize;
        for (std::size_t biasMatrixIndex = biasMatrixBatchOffset;
             biasMatrixIndex < biasMatrixBatchOffset + biasMatrixSize;
             ++biasMatrixIndex)
        {
            const auto matrixIdx = biasMatrixBatchOffset % biasMatrixSize;
            const auto channelIdx = matrixIdx % mapSize;
            bias.At(channelIdx) += biasMatrix.At(biasMatrixIndex);
        }
    }
}
}

#endif
