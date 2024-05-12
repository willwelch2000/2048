using MathNet.Numerics.LinearAlgebra;

namespace AI2048.Deep;

/// <summary>
/// 
/// </summary>
/// <typeparam name="S">Represents type of state</typeparam>
/// <typeparam name="A">Represents type of action</typeparam>
public class DeepQLearner<S, A> : QLearner<S, A>
{
    // fields

    private NeuralNet targetNet;
    private NeuralNet mainNet;
    private int iterationCounter;
    /// <summary>
    /// Number of moves made before the main network is copied to the target network
    /// </summary>
    private const int iterationsBeforeTransfer = 100;
    private Sigmoid sigmoid = new();


    // // // constructors

    public DeepQLearner(IQLearnAgent<S, A> agent) : base(agent)
    {
        targetNet = new(agent.NeuralNetInputLayerSize, 50, agent.OutputSize, 2);
        mainNet = targetNet.Clone();
    }


    // // // methods

    public override A? GetActionFromQValues(S state)
    {
        Vector<double> output = mainNet.GetOutputValues(agent.GetNeuralNetFeatures(state));
        double max = output.Max();
        int[] maxNodes = Enumerable.Range(0, output.Count).Where(x => output[x] >= max).ToArray();

        // Choose randomly
        int randomIndex = random.Next(maxNodes.Length);

        return agent.GetActionFromNodeNumber(maxNodes[randomIndex]);
    }


    public override double GetQValue(S state, A action)
    {
        Vector<double> output = mainNet.GetOutputValues(agent.GetNeuralNetFeatures(state));
        return output[agent.GetNodeNumberFromAction(action)];
    }

    public override void Update(S state, A action, S nextState)
    {
        Vector<double> input = agent.GetNeuralNetFeatures(state);
        double reward = agent.GetReward(state, action, nextState);
        double valueNextState = GetValueFromQValues(nextState);
        Vector<double> output = mainNet.GetOutputValues(input);
        output[agent.GetNodeNumberFromAction(action)] = sigmoid.Activate(reward + agent.Discount * valueNextState);
        mainNet.PerformGradientDescent(input, output);

        if (iterationCounter == iterationsBeforeTransfer)
        {
            iterationCounter = 0;
            targetNet = mainNet;
            mainNet = mainNet.Clone();
        }

        TotalRewards += reward;
    }
}