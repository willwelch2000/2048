using AI2048.Deep;
using Game2048.Display;
using Game2048.Game;
using MathNet.Numerics.LinearAlgebra;

namespace AI2048;

public class Program
{
    public static void Main()
    {
        // RunWithoutTraining(Util.GetNeuralNetFromFile("reluslope_256x2x50_6_28_v7.txt"));
        // RunAndSaveToFile(Util.GetNeuralNetFromFile("reluslope_256x2x50_6_28_v6.txt"), "reluslope_256x2x50_6_28", 3);
        RunAndSaveToFile(null, "reluslope_256x2x50_7_1", 1);
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
        var nodes = net.Nodes.ToArray();
        double h0 = nodes[1][0];
        double h1 = nodes[1][1];
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
        DeepQLearner<int[,], Direction> deepQLearner = new(agent, 50, 2)
        {
            Epsilon = 1,
            EpsilonDecay = 0.99,
            IterationsBeforeNetTransfer = 50,
        };
        IDisplay display = new CommandLineDisplay(agent.Game);
        deepQLearner.PerformQLearning(6);
        Console.WriteLine($"Test score (training): {deepQLearner.AverageScore}");
        Console.WriteLine($"Test rewards (training): {deepQLearner.AverageRewards}");
        double testScore = deepQLearner.AverageScore;
        double testReward = deepQLearner.AverageRewards;

        Console.WriteLine("done training");

        deepQLearner.ResetStats();
        deepQLearner.Epsilon = 0;
        deepQLearner.PerformQLearning(6);
        Console.WriteLine($"Test score (training): {testScore}");
        Console.WriteLine($"Test rewards (training): {testReward}");
        Console.WriteLine($"Test score (post-training): {deepQLearner.AverageScore}");
        Console.WriteLine($"Test rewards (post-training): {deepQLearner.AverageRewards}");
    }

    public static void SaveToFileTest()
    {
        NeuralNet net1 = new(16, 50, 4, 2);
        net1.SaveTrainingState("tempfileneuralnettest.txt");
        NeuralNet net2 = Util.GetNeuralNetFromFile("tempfileneuralnettest.txt");

        // Confirm equality

        // // Weights
        // int weights1Length = net1.Weights.Count();
        // int weights2Length = net2.Weights.Count();
        // for (int i = 0; i < weights1Length; i++)
        // {
        //     Matrix<double> weights1 = net1.Weights.ElementAt(i);
        //     Matrix<double> weights2 = net2.Weights.ElementAt(i);
        // }

        // // Biases
        // int biases1Length = net1.Biases.Count();
        // int biases2Length = net2.Biases.Count();
        // for (int i = 0; i < biases1Length; i++)
        // {
        //     Vector<double> biases1 = net1.Biases.ElementAt(i);
        //     Vector<double> biases2 = net2.Biases.ElementAt(i);
        // }

        // Delete file
        File.Delete("tempfileneuralnettest.txt");
    }

    public static void RunAndSaveToFile(NeuralNet? startingNet, string filename, int start, int iterationsBeforeSave=500, int iterationsBeforeAlphaUpdate=25, int totalIterations=2000)
    {
        IActivationFunction activationFunction = new ReLUWithSlopes(0.1, 0.001);
        Agent2048 agent = new();
        DeepQLearner<int[,], Direction> deepQLearner;
        if (startingNet is not null)
            deepQLearner = new(agent, startingNet);
        else
            deepQLearner = new(agent, 50, 2, activationFunction);
        deepQLearner.Epsilon = 0.2;
        deepQLearner.EpsilonDecay = 0.99;
        deepQLearner.MinEpsilon = 0.1;
        deepQLearner.IterationsBeforeNetTransfer = 1000;
        // _ = new CommandLineDisplay(agent.Game);

        for (int i = 0; i < totalIterations / iterationsBeforeAlphaUpdate; i++)
        {
            // Update alpha based on initial output values from neural network--higher values means alpha needs to be lower to avoid drastic changes
            agent.Restart();
            double max = deepQLearner.MainNet.GetOutputValues(agent.GetNeuralNetFeatures(agent.GetGameState())).AbsoluteMaximum();
            deepQLearner.Alpha = 0.1 / (max == 0 ? 1 : max);

            // Do learning
            deepQLearner.ResetStats();
            deepQLearner.PerformQLearning(iterationsBeforeAlphaUpdate);
            Console.WriteLine($"Test score: {deepQLearner.AverageScore}");
            Console.WriteLine($"Test rewards: {deepQLearner.AverageRewards}");

            // Save state if correct iteration (last in cycle)
            if ((i + 1) % (iterationsBeforeSave / iterationsBeforeAlphaUpdate) == 0)
            {
                deepQLearner.TargetNet.SaveTrainingState($"{filename}_v{start + i/iterationsBeforeAlphaUpdate}.txt");
            }
        }


        // for (int v = start; v < start + 3; v++)
        // {
        //     // Set alpha based on initial output values from neural network--higher values means alpha needs to be lower to avoid drastic changes
        //     agent.Restart();
        //     double max = deepQLearner.MainNet.GetOutputValues(agent.GetNeuralNetFeatures(agent.GetGameState())).AbsoluteMaximum();
        //     deepQLearner.Alpha = 0.1 / (max == 0 ? 1 : max);

        //     deepQLearner.ResetStats();
        //     deepQLearner.PerformQLearning(50);
        //     Console.WriteLine($"Test score: {deepQLearner.AverageScore}");
        //     Console.WriteLine($"Test rewards: {deepQLearner.AverageRewards}");

        //     // Adjust alpha based on average rewards
        //     // deepQLearner.Alpha = 0.1 / deepQLearner.AverageRewards;
        //     deepQLearner.TargetNet.SaveTrainingState($"{filename}_v{v}.txt");
        // }
    }

    public static void RunWithoutTraining(NeuralNet startingNet)
    {
        foreach (var layerTransform in startingNet.LayerTransforms)
            layerTransform.Activator = new ReLUWithSlopes(0.1, 0.001);
        startingNet.LayerTransforms.Last().Activator = new NoActivation();
        Agent2048 agent = new();
        DeepQLearner<int[,], Direction> deepQLearner = new(agent, startingNet);
        _ = new CommandLineDisplay(agent.Game);
        deepQLearner.PerformWithoutTraining(100);
        Console.WriteLine($"Test score: {deepQLearner.AverageScore}");
        Console.WriteLine($"Test rewards: {deepQLearner.AverageRewards}");
    }
}