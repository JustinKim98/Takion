// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "SimpleMnist.hpp"
#include <cubbydnn/Engine/Model.hpp>

#include "cubbydnn/Computations/LossFunctions/LossFunctions.hpp"

namespace CubbyDNN::Test
{
void SimpleMnistTest()
{
    const Compute::Device device(0, Compute::DeviceType::Cpu, " myDevice");
    Graph::Model model(NumberSystem::Float);
    auto id = model.PlaceHolder({ 100, 3, 3 }, "input", device);

    id = model.Dense(id, 10, std::make_unique<XavierNormal>(),
                     std::make_unique<HeNormal>(), "dense1", device);

    id = model.Activation(id, "ReLU", "act1", device);

    auto loss = Compute::MSE<float>();

    model.Compile("MSE", 
        Graph::ParameterPack({},
            { { "epsilon", 0.01f } }, {}));

    //model.Train(100);
}

}