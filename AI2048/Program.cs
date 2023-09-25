using Game2048.Game;
using MathNet.Numerics.LinearAlgebra;

namespace AI2048;

public class Program
{
    public static void Main(string[] args)
    {
        TrainAndTest();
    }

    public static void TrainAndTest()
    {
        int trainEpisodes = 200;
        int testEpisodes = 200;

        IApproximateQLearnAgent<int[,], Direction> agent = new Agent2048();
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
}