/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_BLAZE_HPP
#define CUBBYDNN_BLAZE_HPP
#ifdef WITH_BLAZE

#include <blaze/math/CustomMatrix.h>
#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN::Compute
{
using blaze::columnMajor;
using blaze::CustomMatrix;
using blaze::aligned;
using blaze::padded;
using blaze::unaligned;
using blaze::unpadded;

class Blaze
{
private:
    static std::size_t m_getPaddedSize(std::size_t padSize,
                                       std::size_t numCols)
    {
        if (padSize == 0)
            return numCols;

        std::size_t i = 0;
        while (padSize * i < numCols)
            i++;
        return padSize * i;
    }

public:
    template <typename T>
    static void TensorMul(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSize = inputShapeA.NumMatrices();

        std::size_t colDataSizeA = inputA.GetPaddedNumCols();
        std::size_t colDataSizeB = inputB.GetPaddedNumCols();
        std::size_t colDataSizeOutput = output.GetPaddedNumCols();

        const auto matrixSizeA = inputShapeA.NumRows() * colDataSizeA;
        const auto matrixSizeB = inputShapeB.NumRows() * colDataSizeB;
        const auto matrixSizeOut = outputShape.NumRows() * colDataSizeOutput;

        T* inputPtrA = static_cast<T*>(inputA.DataPtr.get());
        T* inputPtrB = static_cast<T*>(inputB.DataPtr.get());
        T* outputPtr = static_cast<T*>(output.DataPtr.get());

        if (inputA.Device.PadSize() > 0)
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                    inputPtrA + batchIdx * matrixSizeA,
                    inputShapeA.NumRows(), inputShapeA.NumCols(),
                    colDataSizeA);

                const CustomMatrix<T, aligned, padded, blaze::rowMajor> B(
                    inputPtrB + batchIdx * matrixSizeB,
                    inputShapeB.NumRows(), inputShapeB.NumCols(),
                    colDataSizeB);

                CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                    outputPtr + batchIdx * matrixSizeOut,
                    outputShape.NumRows(), outputShape.NumCols(),
                    colDataSizeOutput);

                Out = A * B;
            }
        else
        {
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                    inputPtrA + batchIdx * matrixSizeA,
                    inputShapeA.NumRows(), inputShapeA.NumCols());

                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                    inputPtrB + batchIdx * matrixSizeB,
                    inputShapeB.NumRows(), inputShapeB.NumCols());

                CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                    outputPtr + batchIdx * matrixSizeOut,
                    outputShape.NumRows(), outputShape.NumCols());

                Out = A * B;
            }
        }
    }

    template <typename T>
    static void TensorAdd(const Tensor& inputA, const Tensor& inputB,
                          Tensor& output)
    {
        const auto inputShapeA = inputA.TensorShape;
        const auto inputShapeB = inputB.TensorShape;
        const auto outputShape = output.TensorShape;

        const auto batchSize = inputShapeA.NumMatrices();

        std::size_t colDataSizeA = inputA.GetPaddedNumCols();
        std::size_t colDataSizeB = inputB.GetPaddedNumCols();
        std::size_t colDataSizeOutput = output.GetPaddedNumCols();

        const auto matrixSizeA = inputShapeA.NumRows() * colDataSizeA;
        const auto matrixSizeB = inputShapeB.NumRows() * colDataSizeB;
        const auto matrixSizeOut = outputShape.NumRows() * colDataSizeOutput;

        T* inputPtrA = static_cast<T*>(inputA.DataPtr.get());
        T* inputPtrB = static_cast<T*>(inputB.DataPtr.get());
        T* outputPtr = static_cast<T*>(output.DataPtr.get());

        if (inputA.Device.PadSize() > 0)
        {
            const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                inputPtrA,
                inputShapeA.NumRows() * batchSize, inputShapeA.NumCols(),
                colDataSizeA);

            const CustomMatrix<T, aligned, padded, blaze::rowMajor> B(
                inputPtrB,
                inputShapeB.NumRows() * batchSize, inputShapeB.NumCols(),
                colDataSizeB);

            CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                outputPtr,
                outputShape.NumRows() * batchSize, outputShape.NumCols(),
                colDataSizeOutput);

            Out = A + B;
        }
        else
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                    inputPtrA + batchIdx * matrixSizeA,
                    inputShapeA.NumRows(), inputShapeA.NumCols());

                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> B(
                    inputPtrB + batchIdx * matrixSizeB,
                    inputShapeB.NumRows(), inputShapeB.NumCols());

                CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                    outputPtr + batchIdx * matrixSizeOut,
                    outputShape.NumRows(), outputShape.NumCols());

                Out = A + B;
            }
    }

    template <typename T>
    static void TensorAdd(Tensor& tensor, const Tensor& toAdd)
    {
        const auto inputShapeA = tensor.TensorShape;
        const auto outputShape = toAdd.TensorShape;

        const auto batchSize = inputShapeA.NumMatrices();

        std::size_t colDataSizeA = tensor.GetPaddedNumCols();
        std::size_t colDataSizeOutput = toAdd.GetPaddedNumCols();

        const auto matrixSizeA = inputShapeA.NumRows() * colDataSizeA;
        const auto matrixSizeOut = outputShape.NumRows() * colDataSizeOutput;

        T* inputPtrA = static_cast<T*>(tensor.DataPtr.get());
        T* outputPtr = static_cast<T*>(toAdd.DataPtr.get());

        if (tensor.Device.PadSize() > 0)
        {
            const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                inputPtrA, inputShapeA.NumRows() * batchSize,
                inputShapeA.NumCols(), colDataSizeA);

            CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                outputPtr, outputShape.NumRows() * batchSize,
                outputShape.NumCols(), colDataSizeOutput);

            Out += A;
        }
        else
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                    inputPtrA + batchIdx * matrixSizeA, inputShapeA.NumRows(),
                    inputShapeA.NumCols());

                CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                    outputPtr + batchIdx * matrixSizeOut, outputShape.NumRows(),
                    outputShape.NumCols());

                Out += A;
            }
    }

    template <typename T>
    static void BatchMean(const Tensor& input, std::size_t idx, Tensor& output)
    {
        std::size_t interval = 1;
        for (int i = input.TensorShape.Dim() - 1; i > static_cast<int>(idx);
             --i)
            if (i == input.TensorShape.Dim() - 1)
                interval *= input.GetPaddedNumCols();
            else
                interval *= input.TensorShape.At(i);

        std::size_t batchSize = 1;
        for (std::size_t i = 0; i <= idx; ++i)
            if (i == input.TensorShape.Dim() - 1)
                interval *= input.GetPaddedNumCols();
            else
                interval *= input.TensorShape.At(i);

        const auto colDataSizeA = input.GetPaddedNumCols();
        const auto colDataSizeOutput = output.GetPaddedNumCols();

        const auto inputShape = input.TensorShape;
        const auto outputShape = output.TensorShape;

        const T* inputPtr = static_cast<T*>(input.DataPtr.get());
        T* outputPtr = static_cast<T*>(output.DataPtr.get());

        CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
            outputPtr, outputShape.NumRows() * batchSize, outputShape.NumCols(),
            colDataSizeOutput);

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                inputPtr + batchSize * interval,
                inputShape.NumRows() * batchSize,
                inputShape.NumCols(), colDataSizeA);
        }

        for (std::size_t elementIdx = 0; elementIdx < interval; ++elementIdx)
            outputPtr[elementIdx] /= batchSize;
    }

    template <typename T>
    static void TensorTranspose(const Tensor& input, Tensor& output)
    {
        const auto inputShape = input.TensorShape;
        const auto outputShape = output.TensorShape;

        std::size_t colDataSizeInput = input.GetPaddedNumCols();
        std::size_t colDataSizeOutput = output.GetPaddedNumCols();

        const auto matrixSizeInput = inputShape.NumRows() * colDataSizeInput;
        const auto matrixSizeOutput = outputShape.NumRows() * colDataSizeOutput;
        const auto batchSize = inputShape.NumMatrices();

        T* inputPtr = static_cast<T*>(input.DataPtr.get());
        T* outputPtr = static_cast<T*>(output.DataPtr.get());

        if (input.Device.PadSize() > 0)
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, aligned, padded, blaze::rowMajor> A(
                    inputPtr +
                    batchIdx * matrixSizeInput,
                    inputShape.NumRows(), inputShape.NumCols(),
                    colDataSizeInput);

                CustomMatrix<T, aligned, padded, blaze::rowMajor> Out(
                    outputPtr +
                    batchIdx * matrixSizeOutput,
                    outputShape.NumRows(), outputShape.NumCols(),
                    colDataSizeOutput);

                Out = blaze::trans(A);
            }
        else
            for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                const CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> A(
                    inputPtr +
                    batchIdx * matrixSizeInput,
                    inputShape.NumRows(), inputShape.NumCols()
                    );

                CustomMatrix<T, unaligned, unpadded, blaze::rowMajor> Out(
                    outputPtr +
                    batchIdx * matrixSizeOutput,
                    outputShape.NumRows(), outputShape.NumCols());

                Out = blaze::trans(A);
            }
    }
};
} // namespace CubbyDNN

#endif
#endif
