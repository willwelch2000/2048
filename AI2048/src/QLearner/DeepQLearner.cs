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
    public int IterationsBeforeNetTransfer { get; set; } = 100;


    // // // constructors

    public DeepQLearner(IQLearnAgent<S, A> agent, int numMiddleNodes, int numMiddleLayers, IActivationFunction? activator = null) : base(agent)
    {
        if (activator is null)
            targetNet = new(agent.NeuralNetInputLayerSize, numMiddleNodes, agent.OutputSize, numMiddleLayers);
        else
            targetNet = new(agent.NeuralNetInputLayerSize, numMiddleNodes, agent.OutputSize, numMiddleLayers, activator);
        mainNet = targetNet.Clone();
    }

    public DeepQLearner(IQLearnAgent<S, A> agent, NeuralNet startingNet) : base(agent)
    {
        // Confirm that net has right size
        if (startingNet.NumInputNodes != agent.NeuralNetInputLayerSize || startingNet.NumOutputNodes != agent.OutputSize)
            throw new Exception($"Neural network should have input size of {agent.NeuralNetInputLayerSize} and output size of {agent.OutputSize}");

        targetNet = startingNet.Clone();
        mainNet = startingNet.Clone();
    }


    // // // properties

    public NeuralNet MainNet => mainNet.Clone();

    public NeuralNet TargetNet => targetNet.Clone();

    public override double Alpha
    {
        get => targetNet.Alpha;
        set
        {
            mainNet.Alpha = value;
            targetNet.Alpha = value;
        }
    }


    // // // methods

    public override A? GetActionFromQValues(S state)
    {
        // Get node numbers of legal actions
        int[] legalActionNodeNumbers = agent.GetLegalActions(state).Select(agent.GetNodeNumberFromAction).ToArray();
        if (legalActionNodeNumbers.Length == 0)
            return default;
        
        // Get output from neural net
        Vector<double> output = mainNet.GetOutputValues(agent.GetNeuralNetFeatures(state));

        // Max value of legal actions
        double max = legalActionNodeNumbers.Select(i => output[i]).Max();

        // All node numbers that meet this value--could be multiple that are tied for highest
        int[] maxNodes = legalActionNodeNumbers.Where(i => output[i] >= max).ToArray();

        // If only one, return that action
        if (maxNodes.Length == 1)
            return agent.GetActionFromNodeNumber(maxNodes[0]);

        // Otherwise, choose randomly from the options
        int randomIndex = random.Next(maxNodes.Length);
        return agent.GetActionFromNodeNumber(maxNodes[randomIndex]);
    }


    public override double GetQValue(S state, A action)
    {
        Vector<double> output = mainNet.GetOutputValues(agent.GetNeuralNetFeatures(state));
        return output[agent.GetNodeNumberFromAction(action)];
    }

    public override void Update(S state, A action, S nextState, double reward)
    {
        // Input is neural net features
        Vector<double> input = agent.GetNeuralNetFeatures(state);

        double valueNextState = GetValueFromQValues(nextState);

        // Output used for gradient descent is current output, 
        // but with the node of the taken action changed to (activated): reward + agent.Discount * next state value
        Vector<double> output = mainNet.GetOutputValues(input);
        output[agent.GetNodeNumberFromAction(action)] = mainNet.Activator.Activate(reward + agent.Discount * valueNextState);
        
        targetNet.PerformGradientDescent(input, output);

        // After given number of iterations, replace target net with main net
        if (iterationCounter++ == IterationsBeforeNetTransfer)
        {
            iterationCounter = 0;
            mainNet = targetNet;
            targetNet = mainNet.Clone();
        }

        TotalRewards += reward;
    }
}