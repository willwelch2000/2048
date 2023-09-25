
using AI2048;
using AI2048.Deep;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace Test2048;

[TestClass]
public class NeuralNetTests
{
    [TestMethod]
    public void BasicCalculationTest()
    {
        NeuralNet net = new(2, 2, 2, 1);

        // Set weights

        net.SetWeight(0, 0, 0, 1);
        net.SetWeight(0, 0, 1, 2);
        net.SetWeight(0, 1, 0, -1);
        net.SetWeight(0, 1, 1, -2);
        
        net.SetWeight(1, 0, 0, 5);
        net.SetWeight(1, 0, 1, 6);
        net.SetWeight(1, 1, 0, -5);
        net.SetWeight(1, 1, 1, -6);

        // Set biases

        net.SetBias(0, 0, 3);
        net.SetBias(0, 1, 4);
        
        net.SetBias(1, 0, 7);
        net.SetBias(1, 1, 8);

        Vector<double> input = Vector<double>.Build.Dense(new double[] {0.5, 0.75});
        Vector<double> output = net.GetOutputValues(input);

        double tolerance = 0.0001;
        double output0 = 0.998938;
        double output1 = 0.9995967;
        Assert.IsTrue(output[0] > output0 - tolerance && output[0] < output0 + tolerance);
        Assert.IsTrue(output[1] > output1 - tolerance && output[1] < output1 + tolerance);
    }
}