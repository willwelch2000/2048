namespace AI2048;

/// <summary>
/// Extension of QLearner that uses Approximate Q learning
/// </summary>
/// <typeparam name="S">Represents type of state</typeparam>
/// <typeparam name="A">Represents type of action</typeparam>
public class ApproximateQLearner<S, A>(IQLearnAgent<S, A> agent) : QLearner<S, A>(agent)
{
    // // // properties

    /// <summary>
    /// Dictionary mapping the name of a feature to its value
    /// </summary>
    public Dictionary<string, double> Weights { get; set; } = new();


    // // // methods

    /// <summary>
    /// Get the q-value of a q-state
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <returns>q-value as double</returns>
    public override double GetQValue(S state, A action)
    {
        double qValue = Util.DictionaryDotProduct(agent.GetFeatures(state, action), Weights);
        return qValue;
    }

    /// <summary>
    /// Update weights given a move
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <param name="nextState"></param>
    public override void Update(S state, A action, S nextState, double reward)
    {
        Dictionary<string, double> features = agent.GetFeatures(state, action);
        double valueNextState = GetValueFromQValues(nextState);
        double qValue = GetQValue(state, action);
        double correction = reward + agent.Discount * valueNextState - qValue;
        foreach (string featureName in features.Keys)
            Weights[featureName] = SafeGetWeight(featureName) + Alpha * correction * features[featureName];

        TotalRewards += reward;
    }

    /// <summary>
    /// If the Weight has a value, return that. Otherwise, return 0
    /// </summary>
    /// <param name="featureName"></param>
    /// <returns></returns>
    private double SafeGetWeight(string featureName)
    {
        if (Weights.TryGetValue(featureName, out double value))
            return value;
        return 0;
    }
}