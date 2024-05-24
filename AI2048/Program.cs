using AI2048.Deep;
using Game2048.Display;
using Game2048.Game;
using MathNet.Numerics.LinearAlgebra;

namespace AI2048;

public class Program
{
    public static void Main()
    {
        TrainAndTestDeepQLearning();
    }

    public static void NNTest() {
        
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

        double y1 = output[1];
        double h0 = net.Nodes[1][0];
        double h1 = net.Nodes[1][1];
        double dh0dw010 = 0;
        double dh1dw010 = h1*(1-h1)*input[0];
        double dy0dw010 = y1*(1-y1)*(1*dh0dw010 - 1*dh1dw010);
        Console.WriteLine(dy0dw010);
    }

    public static void TrainAndTest()
    {
        int trainEpisodes = 200;
        int testEpisodes = 200;

        IQLearnAgent<int[,], Direction> agent = new Agent2048();
        ApproximateQLearner<int[,], Direction> approximateQLearner = new(agent)
        {
            Epsilon = 1,
            EpsilonDecay = 0.99
        };
        approximateQLearner.PerformQLearning(trainEpisodes);

        Console.WriteLine("done training");

        approximateQLearner.ResetStats();
        approximateQLearner.Epsilon = 0;
        approximateQLearner.PerformQLearning(testEpisodes);
        Console.WriteLine($"Test score: {approximateQLearner.AverageScore}");
        Console.WriteLine($"Test rewards: {approximateQLearner.AverageRewards}");
    }

    public static void TrainAndTestDeepQLearning()
    {
        Agent2048 agent = new();
        DeepQLearner<int[,], Direction> deepQLearner = new(agent)
        {
            Epsilon = 1,
            EpsilonDecay = 0.99,
        };
        IDisplay display = new CommandLineDisplay(agent.Game);
        deepQLearner.PerformQLearning(100);

        Console.WriteLine("done training");

        deepQLearner.ResetStats();
        deepQLearner.Epsilon = 0;
        deepQLearner.PerformQLearning(100);
        Console.WriteLine($"Test score: {deepQLearner.AverageScore}");
        Console.WriteLine($"Test rewards: {deepQLearner.AverageRewards}");
    }
}