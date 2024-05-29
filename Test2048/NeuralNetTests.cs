
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
    public void TestDerivatives()
    {
        double tolerance = 0.0001;

        NeuralNet net = new(2, 2, 2, 1);

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

        Vector<double> input = Vector<double>.Build.Dense(new double[] {1, 2});
        Vector<double> output = net.GetOutputValues(input);

        double derivH0 = net.NodeDerivative(2, 0, 1, 0);
        double correctDerivH0 = 0.4313;
        Assert.IsTrue(derivH0 > correctDerivH0 - tolerance && derivH0 < correctDerivH0 + tolerance);

        double derivB11 = net.BiasDerivative(2, 0, 1, 0);
        double correctDerivB11 = 0.08626804196599218;
        Assert.IsTrue(derivB11 > correctDerivB11 - tolerance && derivB11 < correctDerivB11 + tolerance);

        double derivW101 = net.WeightDerivative(2, 0, 1, 0, 1);
        double correctDerivW101 = 0.004091337217556154;
        Assert.IsTrue(derivW101 > correctDerivW101 - tolerance && derivW101 < correctDerivW101 + tolerance);

        double derivX0 = net.NodeDerivative(2, 0, 0, 0);
        double correctDerivX0 = 0.19299707694420815;
        Assert.IsTrue(derivX0 > correctDerivX0 - tolerance && derivX0 < correctDerivX0 + tolerance);

        double derivB00 = net.BiasDerivative(2, 1, 0, 0);
        double correctDerivB00 = 0.01082859047691552;
        Assert.IsTrue(derivB00 > correctDerivB00 - tolerance && derivB00 < correctDerivB00 + tolerance);

        double derivW010 = net.WeightDerivative(2, 1, 0, 1, 0);
        double correctDerivW010 = -0.002488147790806601;
        Assert.IsTrue(derivW010 > correctDerivW010 - tolerance && derivW010 < correctDerivW010 + tolerance);
    }

    [TestMethod]
    public void TestGradDescent2Layer1Output()
    {
        double tolerance = 0.01;

        NeuralNet net = new(2, 2, 1, 0);
        net.Alpha = 10;

        // Create comparison network
        NeuralNet compare = new(2, 2, 1, 0);
        compare.SetWeight(0, 0, 0, -5);
        compare.SetWeight(0, 1, 0, 10);
        compare.SetBias(0, 0, -3);

        Random random = new(1000);
        for (int i = 0; i < 3000; i++)
        {
            Vector<double> input = Vector<double>.Build.DenseOfArray(random.NextDoubles(2));
            Vector<double> output = compare.GetOutputValues(input);
            net.PerformGradientDescent(input, output);
        }

        var weights = net.Weights.ToArray();
        var compareWeights = compare.Weights.ToArray();
        for (int i = 0; i < weights.Length; i++)
            for (int j = 0; j < weights[i].RowCount; j++)
                for (int k = 0; k < weights[i].ColumnCount; k++)
                    Assert.IsTrue(weights[i][j, k] > compareWeights[i][j, k] - tolerance && weights[i][j, k] < compareWeights[i][j, k] + tolerance);
    }

    [TestMethod]
    public void SaveNetToFileTest()
    {
        NeuralNet net1 = new(16, 50, 4, 2);
        net1.SaveTrainingState("tempfileneuralnettest.txt");
        NeuralNet net2 = Util.GetNeuralNetFromFile("tempfileneuralnettest.txt");

        // Confirm equality

        // Weights
        int weights1Length = net1.Weights.Count();
        int weights2Length = net2.Weights.Count();
        Assert.AreEqual(weights1Length, weights2Length);
        for (int i = 0; i < weights1Length; i++)
        {
            Matrix<double> weights1 = net1.Weights.ElementAt(i);
            Matrix<double> weights2 = net2.Weights.ElementAt(i);
            Assert.AreEqual(weights1, weights2);
        }

        // Biases
        int biases1Length = net1.Biases.Count();
        int biases2Length = net2.Biases.Count();
        Assert.AreEqual(biases1Length, biases2Length);
        for (int i = 0; i < biases1Length; i++)
        {
            Vector<double> biases1 = net1.Biases.ElementAt(i);
            Vector<double> biases2 = net2.Biases.ElementAt(i);
            Assert.AreEqual(biases1, biases2);
        }

        // Delete file
        File.Delete("tempfileneuralnettest.txt");
    }
}