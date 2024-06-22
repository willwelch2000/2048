
using AI2048;
using AI2048.Deep;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;

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

    [TestMethod]
    public void TestToNodeDerivatives()
    {
        double tolerance = 0.0001;

        NeuralNet net = GetNeuralNet();

        Vector<double> input = Vector<double>.Build.Dense([1, 2]);
        Vector<double> output = net.GetOutputValues(input);
        
        double derivH0 = net.GetNodeToNodeDerivative(2, 0, 1, 0);
        double correctDerivH0 = 0.4313;
        Assert.IsTrue(derivH0 > correctDerivH0 - tolerance && derivH0 < correctDerivH0 + tolerance);

        double derivB11 = net.GetNodeToBiasDerivative(2, 0, 1, 0);
        double correctDerivB11 = 0.08626804196599218;
        Assert.IsTrue(derivB11 > correctDerivB11 - tolerance && derivB11 < correctDerivB11 + tolerance);

        double derivW101 = net.GetNodeToWeightDerivative(2, 0, 1, 0, 1);
        double correctDerivW101 = 0.004091337217556154;
        Assert.IsTrue(derivW101 > correctDerivW101 - tolerance && derivW101 < correctDerivW101 + tolerance);

        double derivX0 = net.GetNodeToNodeDerivative(2, 0, 0, 0);
        double correctDerivX0 = 0.19299707694420815;
        Assert.IsTrue(derivX0 > correctDerivX0 - tolerance && derivX0 < correctDerivX0 + tolerance);

        double derivB00 = net.GetNodeToBiasDerivative(2, 1, 0, 0);
        double correctDerivB00 = 0.01082859047691552;
        Assert.IsTrue(derivB00 > correctDerivB00 - tolerance && derivB00 < correctDerivB00 + tolerance);

        double derivW010 = net.GetNodeToWeightDerivative(2, 1, 0, 1, 0);
        double correctDerivW010 = -0.002488147790806601;
        Assert.IsTrue(derivW010 > correctDerivW010 - tolerance && derivW010 < correctDerivW010 + tolerance);
    }

    [TestMethod]
    public void TestLossDerivatives()
    {
        double tolerance = 0.0001;

        NeuralNet net = GetNeuralNet();

        Vector<double> input = Vector<double>.Build.Dense([1, 2]);
        Vector<double> compare = Vector<double>.Build.Dense([0.95, 0.1]);

        net.GetOutputValues(input);
        net.CalculateLossDerivatives(input, compare);

        double derivW100 = net.GetLossToWeightDerivative(1, 0, 0) ?? 0;
        double correctDerivW100 = -0.0907 * net.GetNodeToWeightDerivative(2, 0, 1, 0, 0) + -0.0830 * net.GetNodeToWeightDerivative(2, 1, 1, 0, 0);
        Assert.IsTrue(derivW100 > correctDerivW100 - tolerance && derivW100 < correctDerivW100 + tolerance);

        double derivW110 = net.GetLossToWeightDerivative(1, 1, 0) ?? 0;
        double correctDerivW110 = -0.0907 * net.GetNodeToWeightDerivative(2, 0, 1, 1, 0) + -0.0830 * net.GetNodeToWeightDerivative(2, 1, 1, 1, 0);
        Assert.IsTrue(derivW110 > correctDerivW110 - tolerance && derivW110 < correctDerivW110 + tolerance);

        double derivW101 = net.GetLossToWeightDerivative(1, 0, 1) ?? 0;
        double correctDerivW101 = -0.0907 * net.GetNodeToWeightDerivative(2, 0, 1, 0, 1) + -0.0830 * net.GetNodeToWeightDerivative(2, 1, 1, 0, 1);
        Assert.IsTrue(derivW101 > correctDerivW101 - tolerance && derivW101 < correctDerivW101 + tolerance);

        double derivW111 = net.GetLossToWeightDerivative(1, 1, 1) ?? 0;
        double correctDerivW111 = -0.0907 * net.GetNodeToWeightDerivative(2, 0, 1, 1, 1) + -0.0830 * net.GetNodeToWeightDerivative(2, 1, 1, 1, 1);
        Assert.IsTrue(derivW111 > correctDerivW111 - tolerance && derivW111 < correctDerivW111 + tolerance);

        double derivW000 = net.GetLossToWeightDerivative(0, 0, 0) ?? 0;
        double correctDerivW000 = -0.0907 * net.GetNodeToWeightDerivative(2, 0, 0, 0, 0) + -0.0830 * net.GetNodeToWeightDerivative(2, 1, 0, 0, 0);
        Assert.IsTrue(derivW000 > correctDerivW000 - tolerance && derivW000 < correctDerivW000 + tolerance);

        double derivW010 = net.GetLossToWeightDerivative(0, 1, 0) ?? 0;
        double correctDerivW010 = -0.0907 * net.GetNodeToWeightDerivative(2, 0, 0, 1, 0) + -0.0830 * net.GetNodeToWeightDerivative(2, 1, 0, 1, 0);
        Assert.IsTrue(derivW010 > correctDerivW010 - tolerance && derivW010 < correctDerivW010 + tolerance);

        double derivW001 = net.GetLossToWeightDerivative(0, 0, 1) ?? 0;
        double correctDerivW001 = -0.0907 * net.GetNodeToWeightDerivative(2, 0, 0, 0, 1) + -0.0830 * net.GetNodeToWeightDerivative(2, 1, 0, 0, 1);
        Assert.IsTrue(derivW001 > correctDerivW001 - tolerance && derivW001 < correctDerivW001 + tolerance);

        double derivW011 = net.GetLossToWeightDerivative(0, 1, 1) ?? 0;
        double correctDerivW011 = -0.0907 * net.GetNodeToWeightDerivative(2, 0, 0, 1, 1) + -0.0830 * net.GetNodeToWeightDerivative(2, 1, 0, 1, 1);
        Assert.IsTrue(derivW011 > correctDerivW011 - tolerance && derivW011 < correctDerivW011 + tolerance);
    }

    [TestMethod]
    public void TestGradDescent2Layer1Output()
    {
        int randomSeed = 500;
        double tolerance = 0.5;

        NeuralNet net = new(2, 2, 1, 0, new NoActivation(), randomSeed)
        {
            Alpha = 0.1
        };

        // Create comparison network
        NeuralNet compare = new(2, 2, 1, 0, new NoActivation());
        compare.SetWeight(0, 0, 0, -5);
        compare.SetWeight(0, 1, 0, 10);
        compare.SetBias(0, 0, -3);

        Random random = new(randomSeed);
        for (int i = 0; i < 3000; i++)
        {
            Vector<double> input = Vector<double>.Build.DenseOfArray(random.NextDoubles(2));
            Vector<double> output = compare.GetOutputValues(input);
            net.PerformGradientDescent(input, output);
        }

        Assert.AreEqual(net.LayerTransforms.Count(), compare.LayerTransforms.Count());

        for (int i = 0; i < net.LayerTransforms.Count(); i++)
        {
            var weights = net.LayerTransforms.ElementAt(i).Weights;
            var compareWeights = compare.LayerTransforms.ElementAt(i).Weights;
            for (int j = 0; j < weights.RowCount; j++)
                for (int k = 0; k < weights.ColumnCount; k++)
                    Assert.IsTrue(weights[j, k] > compareWeights[j, k] - tolerance && weights[j, k] < compareWeights[j, k] + tolerance);
        }
    }

    [TestMethod]
    public void SaveNetToFileTest()
    {
        NeuralNet net1 = new(16, 50, 4, 2);
        net1.SaveTrainingState("tempfileneuralnettest.txt");
        NeuralNet net2 = Util.GetNeuralNetFromFile("tempfileneuralnettest.txt");

        // Confirm equality

        // Weights
        int layerTransforms1Length = net1.LayerTransforms.Count();
        int layerTransforms2Length = net2.LayerTransforms.Count();
        Assert.AreEqual(layerTransforms1Length, layerTransforms2Length);
        for (int i = 0; i < layerTransforms1Length; i++)
        {
            Matrix<double> weights1 = net1.LayerTransforms.ElementAt(i).Weights;
            Matrix<double> weights2 = net2.LayerTransforms.ElementAt(i).Weights;
            Assert.AreEqual(weights1, weights2);
        }

        // Biases
        for (int i = 0; i < layerTransforms1Length; i++)
        {
            Vector<double> biases1 = net1.LayerTransforms.ElementAt(i).Biases;
            Vector<double> biases2 = net2.LayerTransforms.ElementAt(i).Biases;
            Assert.AreEqual(biases1, biases2);
        }

        // Delete file
        File.Delete("tempfileneuralnettest.txt");
    }

    private static NeuralNet GetNeuralNet()
    {
        NeuralNet net = new(2, 2, 2, 1, new Sigmoid());

        // Set weights

        net.SetWeight(0, 0, 0, 2);
        net.SetWeight(0, 0, 1, -3);
        net.SetWeight(0, 1, 0, -1);
        net.SetWeight(0, 1, 1, 1);
        
        net.SetWeight(1, 0, 0, 5);
        net.SetWeight(1, 0, 1, 1);
        net.SetWeight(1, 1, 0, -2);
        net.SetWeight(1, 1, 1, -1);

        // Set biases

        net.SetBias(0, 0, -1);
        net.SetBias(0, 1, -2);
        
        net.SetBias(1, 0, 1);
        net.SetBias(1, 1, -3);

        return net;
    }
}