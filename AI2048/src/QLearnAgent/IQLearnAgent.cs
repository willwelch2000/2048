using MathNet.Numerics.LinearAlgebra;

namespace AI2048;

/// <typeparam name="S">Represents type of state</typeparam>
/// <typeparam name="A">Represents type of action</typeparam>
public interface IQLearnAgent<S, A>
{
    // // // properties

    /// <summary>
    /// How much rewards decay (1 is no decay, 0 is full decay)
    /// </summary>
    public double Discount { get; }
    public int InputSize { get; }
    public int OutputSize { get; }
    public int NeuralNetInputLayerSize { get; }


    // // // methods

    /// <summary>
    /// Reset the game
    /// </summary>
    public void Restart();

    /// <summary>
    /// Get the current state of the game
    /// </summary>
    /// <returns></returns>
    public S GetGameState();

    /// <summary>
    /// Do the specified action on the game
    /// </summary>
    /// <param name="action"></param>
    public void PerformAction(A action);

    // methods that should be used like they're static, although they can't be made static using an interface

    /// <summary>
    /// Get all the features of a q-state, defined by a state and action.
    /// Features are named by a string, and their value is a double.
    /// Basically a static function
    /// Used for approximate q learning
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <returns>Dictionary mapping name (string) to value (double) of each feature</returns>
    public Dictionary<string, double> GetFeatures(S state, A action);

    /// <summary>
    /// This becomes the input layer of a neural network.
    /// It's based just off of state, not q-state
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    public Vector<double> GetNeuralNetFeatures(S state);

    /// <summary>
    /// Get list of all legal actions at a state.
    /// Basically static
    /// </summary>
    /// <param name="state"></param>
    /// <returns>list of all legal actions</returns>
    public IEnumerable<A> GetLegalActions(S state);

    /// <summary>
    /// Calculate reward going from state to nextState via action.
    /// Basically static
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <param name="nextState"></param>
    /// <returns></returns>
    public double GetReward(S state, A action, S nextState);

    /// <summary>
    /// Test if a state is a terminal state.
    /// Basically static
    /// </summary>
    /// <param name="state"></param>
    /// <returns>true if terminal</returns>
    public bool IsTerminal(S state);

    /// <summary>
    /// Get score based solely on state. Only used for analytics, not for q learning
    /// May not always be possible, might need to revise this.
    /// But it's useful for 2048
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    public double GetScore(S state);

    /// <summary>
    /// Given a number less than the number of output nodes, get an action.
    /// Used to convert an output node into an action
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    public A GetActionFromNodeNumber(int node);

    /// <summary>
    /// Given an action, convert it into an output node index
    /// </summary>
    /// <param name="action"></param>
    /// <returns></returns>
    public int GetNodeNumberFromAction(A action);
}