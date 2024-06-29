using MathNet.Numerics.LinearAlgebra;

namespace AI2048.Deep;

/// <summary>
/// Extension of QLearner that uses a neural network to approximate the Q-function
/// </summary>
/// <typeparam name="S">Represents type of state</typeparam>
/// <typeparam name="A">Represents type of action</typeparam>
public class DeepQLearner<S, A> : QLearner<S, A>
{
    // fields

    private int iterationCounter;


    // // // constructors

    public DeepQLearner(IQLearnAgent<S, A> agent, int numMiddleNodes, int numMiddleLayers, IActivationFunction? activator = null) : base(agent)
    {
        if (activator is null)
            TargetNet = new(agent.NeuralNetInputLayerSize, numMiddleNodes, agent.OutputSize, numMiddleLayers);
        else
            TargetNet = new(agent.NeuralNetInputLayerSize, numMiddleNodes, agent.OutputSize, numMiddleLayers, activator);
        
        // Default to no activation on output layer
        TargetNet.LayerTransforms.Last().Activator = new NoActivation();
        MainNet = TargetNet.Clone();
    }

    public DeepQLearner(IQLearnAgent<S, A> agent, NeuralNet startingNet) : base(agent)
    {
        // Confirm that net has right size
        if (startingNet.NumInputNodes != agent.NeuralNetInputLayerSize || startingNet.NumOutputNodes != agent.OutputSize)
            throw new Exception($"Neural network should have input size of {agent.NeuralNetInputLayerSize} and output size of {agent.OutputSize}");

        TargetNet = startingNet.Clone();
        MainNet = startingNet.Clone();
    }


    // // // properties

    /// <summary>
    /// Number of moves made before the main network is copied to the target network
    /// </summary>
    public int IterationsBeforeNetTransfer { get; set; } = 100;

    /// <summary>
    /// network that is used for training as a reference
    /// gets replaced with the target network after several iterations
    /// </summary>
    public NeuralNet MainNet { get; private set; }

    /// <summary>
    /// network that gets changed by training
    /// </summary>
    public NeuralNet TargetNet { get; private set; }

    /// <summary>
    /// Alpha is linked to the networks' alpha values
    /// </summary>
    public override double Alpha
    {
        get => TargetNet.Alpha;
        set
        {
            MainNet.Alpha = value;
            TargetNet.Alpha = value;
        }
    }


    // // // methods

    /// <summary>
    /// More efficient way to get action from max q value
    /// </summary>
    /// <param name="state"></param>
    /// <returns>value</returns>
    public override A? GetActionFromQValues(S state)
    {
        // Get node numbers of legal actions
        int[] legalActionNodeNumbers = agent.GetLegalActions(state).Select(agent.GetNodeNumberFromAction).ToArray();
        if (legalActionNodeNumbers.Length == 0)
            return default;
        
        // Get output from neural net
        Vector<double> output = MainNet.GetOutputValues(agent.GetNeuralNetFeatures(state));

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
        Vector<double> output = MainNet.GetOutputValues(agent.GetNeuralNetFeatures(state));
        return output[agent.GetNodeNumberFromAction(action)];
    }

    /// <summary>
    /// More efficient way to get max q value
    /// </summary>
    /// <param name="state"></param>
    /// <returns>value</returns>
    public override double GetValueFromQValues(S state)
    {
        // Get node numbers of legal actions
        int[] legalActionNodeNumbers = agent.GetLegalActions(state).Select(agent.GetNodeNumberFromAction).ToArray();
        if (legalActionNodeNumbers.Length == 0)
            return 0;

        // Get output from neural net
        Vector<double> output = MainNet.GetOutputValues(agent.GetNeuralNetFeatures(state));

        // Max value of legal actions
        return legalActionNodeNumbers.Select(i => output[i]).Max();
    }

    public override void Update(S state, A action, S nextState, double reward)
    {
        // Input is neural net features
        Vector<double> input = agent.GetNeuralNetFeatures(state);

        double valueNextState = GetValueFromQValues(nextState);

        // Output used for gradient descent is current output, 
        // but with the node of the taken action changed to desired Q-value: reward + agent.Discount * next state value
        // Should below be main net or target net
        Vector<double> output = TargetNet.GetOutputValues(input);
        output[agent.GetNodeNumberFromAction(action)] = reward + agent.Discount * valueNextState;
        
        TargetNet.PerformGradientDescent(input, output);

        // After given number of iterations, replace target net with main net
        if (++iterationCounter == IterationsBeforeNetTransfer)
        {
            iterationCounter = 0;
            MainNet = TargetNet.Clone();
        }
    }

    public void SetActivator(int layer, IActivationFunction activator)
    {
        MainNet.SetActivator(layer, activator);
        TargetNet.SetActivator(layer, activator);
    }
}