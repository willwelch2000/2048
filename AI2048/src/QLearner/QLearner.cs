namespace AI2048;

/// <summary>
/// 
/// </summary>
/// <typeparam name="S">Represents type of state</typeparam>
/// <typeparam name="A">Represents type of action</typeparam>
public abstract class QLearner<S, A>(IQLearnAgent<S, A> agent)
{
    // // // fields

    /// <summary>
    /// Object containing details of the game we're learning
    /// </summary>
    protected readonly IQLearnAgent<S, A> agent = agent;

    protected readonly Random random = new();


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
    /// Minimum deviation factor. Epsilon won't decrease below this
    /// </summary>
    public double MinEpsilon { get; set; } = 0;

    /// <summary>
    /// Summed score across all episodes played so far
    /// </summary>
    public double TotalScore { get; protected set; } = 0;

    /// <summary>
    /// Total number of episodes (games) completed so far
    /// </summary>
    public int EpisodesCompleted { get; protected set; } = 0;

    /// <summary>
    /// Average score across all episodes
    /// </summary>
    public double AverageScore => TotalScore / EpisodesCompleted;

    public double TotalRewards { get; protected set; } = 0;
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
        if (legalActions.Length == 0)
            return default;

        if (random.NextDouble() < Epsilon)
        {
            // Choose randomly
            int randomIndex = random.Next(legalActions.Length);
            return legalActions[randomIndex];
        }
        else
            // Follow policy
            return GetActionFromQValues(state);
    }

    /// <summary>
    /// Get the q-value of a q-state
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <returns>q-value as double</returns>
    public abstract double GetQValue(S state, A action);

    /// <summary>
    /// Find the action that results in the q-state with the highest q-value.
    /// If multiple are tied, choose randomly from those
    /// </summary>
    /// <param name="state">current state</param>
    /// <returns>best action, or default if no actions available</returns>
    public virtual A? GetActionFromQValues(S state)
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
    public abstract void Update(S state, A action, S nextState);

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
            if (Epsilon != MinEpsilon)
            {
                double newValEpsilon = Epsilon * EpsilonDecay;
                if (newValEpsilon > MinEpsilon)
                    Epsilon = newValEpsilon;
                else
                    Epsilon = MinEpsilon;
            }
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
}