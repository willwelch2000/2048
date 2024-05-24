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

    /// <summary>
    /// network that gets changed by training
    /// </summary>
    private NeuralNet targetNet;

    /// <summary>
    /// network that is used for training as a reference
    /// gets replaced with the target network after several iterations
    /// </summary>
    private NeuralNet mainNet;

    private int iterationCounter;

    /// <summary>
    /// Number of moves made before the main network is copied to the target network
    /// </summary>
    private const int iterationsBeforeTransfer = 100;
    
    private readonly Sigmoid sigmoid = new();


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
        // Input is neural net features
        Vector<double> input = agent.GetNeuralNetFeatures(state);

        double reward = agent.GetReward(state, action, nextState);
        double valueNextState = GetValueFromQValues(nextState);

        // Output used for gradient descent is current output, 
        // but with the node of the taken action changed to: reward + agent.Discount * next state value
        Vector<double> output = mainNet.GetOutputValues(input);
        output[agent.GetNodeNumberFromAction(action)] = sigmoid.Activate(reward + agent.Discount * valueNextState);
        
        targetNet.PerformGradientDescent(input, output);

        // After given number of iterations, replace target net with main net
        if (iterationCounter++ == iterationsBeforeTransfer)
        {
            iterationCounter = 0;
            mainNet = targetNet;
            targetNet = mainNet.Clone();
        }

        TotalRewards += reward;
    }
}