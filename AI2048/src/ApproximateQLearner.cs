namespace AI2048;

/// <summary>
/// 
/// </summary>
/// <typeparam name="S">Represents type of state</typeparam>
/// <typeparam name="A">Represents type of action</typeparam>
public class ApproximateQLearner<S, A>
{
    // // // fields

    /// <summary>
    /// Object containing details of the game we're learning
    /// </summary>
    private readonly IApproximateQLearnAgent<S, A> agent;

    private readonly Random random = new();


    // // // constructors

    public ApproximateQLearner(IApproximateQLearnAgent<S, A> agent)
    {
        this.agent = agent;
    }


    // // // properties

    /// <summary>
    /// Learning factor. How quickly Q value changes.
    /// In range [0, 1]
    /// </summary>
    public double Alpha { get; set; } = 0.004;

    /// <summary>
    /// Starting deviation factor. Percentage that we act randomly.
    /// In range [0, 1]
    /// </summary>
    public double Epsilon { get; set; } = 1;

    /// <summary>
    /// Epsilon is multiplied by this factor after each episode
    /// </summary>
    public double EpsilonDecay { get; set; } = 0.9;

    /// <summary>
    /// Dictionary mapping the name of a feature to its value
    /// </summary>
    public Dictionary<string, double> Weights { get; set; } = new();

    /// <summary>
    /// Summed score across all episodes played so far
    /// </summary>
    public double TotalScore { get; private set; } = 0;

    /// <summary>
    /// Total number of episodes (games) completed so far
    /// </summary>
    public int EpisodesCompleted { get; private set; } = 0;

    /// <summary>
    /// Average score across all episodes
    /// </summary>
    public double AverageScore => TotalScore / EpisodesCompleted;

    public double TotalRewards { get; private set; } = 0;
    public double AverageRewards => TotalRewards / EpisodesCompleted;


    // // // methods

    /// <summary>
    /// Follow policy (1 - epsilon) of the time.
    /// Choose a random action epsilon of the time.
    /// </summary>
    /// <returns>The action to make, or default if no actions available</returns>
    public A? GetAction(S state)
    {
        A[] legalActions = agent.GetLegalActions(state).ToArray();
        if (!legalActions.Any())
            return default;

        if (random.NextDouble() < Epsilon)
        {
            // Choose randomly
            int randomIndex = random.Next(legalActions.Length);
            return legalActions[randomIndex];
        }
        else
        {
            // Follow policy
            return GetActionFromQValues(state);
        }
    }

    /// <summary>
    /// Get the q-value of a q-state
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <returns>q-value as double</returns>
    public double GetQValue(S state, A action)
    {
        double qValue = Util.DictionaryDotProduct(agent.GetFeatures(state, action), Weights);
        return qValue;
    }

    /// <summary>
    /// Find the action that results in the q-state with the highest q-value.
    /// If multiple are tied, choose randomly from those
    /// </summary>
    /// <param name="state">current state</param>
    /// <returns>best action, or default if no actions available</returns>
    public A? GetActionFromQValues(S state)
    {
        IEnumerable<A> legalActions = agent.GetLegalActions(state);
        if (!legalActions.Any())
            return default;

        // Get list of best actions
        double[] qValues = legalActions.Select(action => GetQValue(state, action)).ToArray();
        double max = qValues.Max();
        A[] bestActions = legalActions.Where(action => GetQValue(state, action) == max).ToArray();

        // Choose randomly
        int randomIndex = random.Next(bestActions.Length);

        return bestActions[randomIndex];
    }

    /// <summary>
    /// Get the value of a state--the max of all its q-values.
    /// Value is 0 if no actions available
    /// </summary>
    /// <param name="state"></param>
    /// <returns>value</returns>
    public double GetValueFromQValues(S state)
    {
        IEnumerable<A> actions = agent.GetLegalActions(state);
        if (!actions.Any())
            return 0;
        return actions.Select(action => GetQValue(state, action)).Max();
    }

    /// <summary>
    /// Update weights given a move
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <param name="nextState"></param>
    public void Update(S state, A action, S nextState)
    {
        Dictionary<string, double> features = agent.GetFeatures(state, action);
        double reward = agent.GetReward(state, action, nextState);
        double valueNextState = GetValueFromQValues(nextState);
        double qValue = GetQValue(state, action);
        double correction = reward + agent.Discount * valueNextState - qValue;
        foreach (string featureName in features.Keys)
            Weights[featureName] = SafeGetWeight(featureName) + Alpha * correction * features[featureName];

        TotalRewards += reward;
    }

    /// <summary>
    /// Iterate through episodes, updating weights and score counters
    /// </summary>
    /// <param name="episodes">how many games to play</param>
    public void PerformQLearning(int episodes)
    {
        // Iterate x episodes
        for (int episodeNum = 0; episodeNum < episodes; episodeNum++)
        {
            agent.Restart();
            S state = agent.GetGameState();

            // Continue until game is over
            while (!agent.IsTerminal(state))
            {
                // Choose a move
                A? nullableAction = GetAction(state);
                // Move shouldn't ever be null--otherwise it would be terminal--but just in case
                if (nullableAction is A action)
                {
                    // Do the action and get updated state
                    agent.PerformAction(action);
                    S nextState = agent.GetGameState();
                    // Update the weights accordingly
                    Update(state, action, nextState);

                    state = nextState;
                }
                else
                    break;
            }

            // Update score variables
            TotalScore += agent.GetScore(state);
            EpisodesCompleted++;
            
            // Update epsilon
            Epsilon *= EpsilonDecay;
        }
    }

    /// <summary>
    /// Set the statistics to 0
    /// </summary>
    public void ResetStats()
    {
        TotalScore = 0;
        EpisodesCompleted = 0;
        TotalRewards = 0;
    }

    /// <summary>
    /// If the Weight has a value, return that. Otherwise, return 0
    /// </summary>
    /// <param name="featureName"></param>
    /// <returns></returns>
    private double SafeGetWeight(string featureName)
    {
        if (Weights.ContainsKey(featureName))
            return Weights[featureName];
        return 0;
    }
}