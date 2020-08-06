// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef TAKION_GRAPH_MODEL_DECL_HPP
#define TAKION_GRAPH_MODEL_DECL_HPP

#include <Takion/Computations/Initializers/InitializerType.hpp>
#include <Takion/Engine/UnitManager.hpp>

#include <Takion/Computations/LossFunctions/LossFunctions.hpp>

namespace Takion::Graph
{
//! Singleton class for maintaining threads that execute the program
template <typename T>
class Model
{
public:
    Model();

    UnitId DataLoader(const Shape& shape,
                      const std::string& name, Compute::Device device);

    //! \param input : unit ID of previous unit
    //! \param units : size of output perceptrons
    //! \param weightInitializer : initializer of the kerne
    //! \param biasInitializer : initializer of the bias
    //! \param name : name of this unit
    //! \param device : device to execute this unit
    UnitId Dense(const UnitId& input, std::size_t units,
                 std::unique_ptr<Initializer<T>> weightInitializer,
                 std::unique_ptr<Initializer<T>> biasInitializer,
                 const std::string& name,
                 Compute::Device device);

    UnitId Dropout(const UnitId& input, float keepRate);

    UnitId Activation(const UnitId& input, const std::string& activationName,
                      const std::string& name, Compute::Device device);

    UnitId Reshape(const UnitId& input, const Shape& shape,
                   const std::string& name = "ReshapeUnit");

    UnitId Loss(const UnitId& prediction, const UnitId& label,
                std::string lossType,
                const std::string& name, Compute::Device device);

    UnitId Loss(
        const UnitId& prediction, const UnitId& label, const std::string& name,
        Compute::Device device,
        std::unique_ptr<Compute::BaseLoss<float>> customLoss);

    UnitId Constant(Tensor tensor, const std::string& name);

    UnitId Multiply(const UnitId& inputA, const UnitId& inputB);

    UnitId Add(const UnitId& inputA, const UnitId& inputB);

    //! OptimizerType, BaseLoss function
    void Compile(const std::string& optimizer,
                 Parameter optimizerParams) noexcept;

    //! Trains the graph with given optimizer and loss function
    void Train(std::size_t epochs, bool async = false);

    void Predict();

    void Predict(void* input, void* output, int workers);

private:

    UnitManager<T> m_unitManager;

    std::size_t m_id = 0;
};
} // namespace Takion

#endif  // CAPTAIN_THREADMANAGER_HPP