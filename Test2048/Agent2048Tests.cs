using MathNet.Numerics.LinearAlgebra;
using AI2048;

namespace Test2048;

[TestClass]
public class Agent2048Tests
{
    [TestMethod]
    public void NeuralNetFeatureExtractorTest()
    {
        int[,] state = new int[4,4] {
            {2, 0, 0, 4},
            {2, 16, 8, 4},
            {4, 32, 64, 0},
            {2, 4, 8, 4}
        };

        Agent2048 agent = new();
        Vector<double> features = agent.GetNeuralNetFeatures(state);

        Assert.AreEqual(features[0], (double) 2 / 10000);
        Assert.AreEqual(features[5], (double) 16 / 10000);
        // Assert.AreEqual(features[16 + 0], 0);
        // Assert.AreEqual(features[16 + 2], 0);
        // Assert.AreEqual(features[16 + 3], 1);
    }
}