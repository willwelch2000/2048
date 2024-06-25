using MathNet.Numerics.LinearAlgebra;
using Game2048.Game;

namespace AI2048;

public class Agent2048 : IQLearnAgent<int[,], Direction>
{
    // // // properties

    public IGame Game { get; private set; } = new Game();

    public double Discount => 1.0;
    public int InputSize => 16;
    public int OutputSize => 4;
    public int NeuralNetInputLayerSize => 256; // Can change to 16


    // // // methods

    public Dictionary<string, double> GetFeatures(int[,] state, Direction action)
    {
        IGame game = new Game(state);
        game.ActionNoAddTile(action);
        int[,] nextState = game.Board;
        int[] stateArr = FlattenIntMatrix(state).ToArray();
        int[] nextStateArr = FlattenIntMatrix(nextState).ToArray();

        Dictionary<string, double> features = new();
        features["reward"] = GetReward(state, action, nextState) / 500;
        // features["increases highest tile (boolean)"] = IncreasesHighestTile(state, action) ? 0.1 : 0;
        // features["total combined"] = ValueCombined(stateArr, nextStateArr) / 500; // certified good
        // features["max tile"] = nextStateArr.Max() / 500; // certified good
        // features["highest in corner"] = HighestInCorner(nextState) ? 0.1 : 0;
        // features["highest in middle"] = HighestInMiddle(nextState) ? 0.1 : 0;
        // features["bias"] = 0.1;
        return features;
    }

    public Vector<double> GetNeuralNetFeatures(int[,] state) => GetNeuralNetFeatures256(state);

    /// <summary>
    /// 16 features representing tile numbers
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    private Vector<double> GetNeuralNetFeatures16(int[,] state)
    {
        Vector<double> features = Vector<double>.Build.Dense(NeuralNetInputLayerSize);
        int i = 0;

        // All tile values
        foreach (int val in state)
            features[i++] = (double) val / 10000;
        
        return features;
    }

    /// <summary>
    /// 16 features representing tile numbers, and 240 representing which (nonzero) tiles are the same
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    private Vector<double> GetNeuralNetFeatures256(int[,] state)
    {
        int[] tiles = FlattenIntMatrix(state).ToArray();
        Vector<double> features = Vector<double>.Build.Dense(NeuralNetInputLayerSize);
        int i = 0;

        // Basic features--all tile values
        foreach (int val in state)
            features[i++] = (double) val / 100;
        
        // Maps of which tiles are the same--if the same, the tile is the value / 100, otherwise -0.1
        for (int j = 0; j < 16; j++)
            for (int k = 0; k < 16; k++)
                if (j != k)
                    features[i++] = tiles[j] == tiles[k] && tiles[j] != 0 ? tiles[k] / 100 : -0.1;
        
        return features;
    }

    public int[,] GetGameState() =>
        Game.Board;

    public IEnumerable<Direction> GetLegalActions(int[,] state)
    {
        foreach (Direction direction in Enum.GetValues(typeof(Direction)))
        {
            Game copy = new(state);
            copy.Action(direction);
            if (!Game.Equals(copy))
                yield return direction;
        }
    }

    /// <summary>
    /// Returns the value of tiles that were combined.
    /// For example, if two 2s combine to form a 4, this returns 4
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <param name="nextState"></param>
    /// <returns></returns>
    public double GetReward(int[,] state, Direction action, int[,] nextState)
    {
        int[] stateArr = FlattenIntMatrix(state).ToArray();
        int[] nextStateArr = FlattenIntMatrix(nextState).ToArray();

        Dictionary<int, int> originalValues = [];
        Dictionary<int, int> nextValues = [];
        Dictionary<int, int> added = [];

        // figure out which number was added as the next tile
        int addedTile = nextStateArr.Sum() - stateArr.Sum();

        // count values in original state
        foreach(int i in stateArr)
            if (i > 0 && !originalValues.TryAdd(i, 1))
                originalValues[i] += 1;

        // count values in next state
        foreach(int i in nextStateArr)
            if (i > 0 && !nextValues.TryAdd(i, 1))
                nextValues[i] += 1;

        // for every number present, figure out the difference between next state and original state
        // i.e. the added tiles of that value
        int nextStateMax = nextValues.Keys.Max();
        int value = nextStateMax;
        added[value] = 0; // initialize highest value dictionary entry
        added[addedTile] = -1; // remove an entry from the value that was added as the additional tile--the loop will cancel out with this
        while(value > 1)
        {
            added.TryAdd(value/2, 0); // initialize the next value's entry
            added[value] += TryGetDictionaryValueOrZero(nextValues, value) - TryGetDictionaryValueOrZero(originalValues, value);
            added[value/2] += 2 * added[value]; // do this to cancel out with the fact that there's two less of a number whenever two combine
            value /= 2; // move to next value
        }

        // return sum of key * number of additions
        return added.Keys.Select(k => k * added[k]).Sum();
    }

    private double OldRewardMethod(int[,] state, Direction action, int[,] nextState)
    {
        int[] flattenedState = FlattenIntMatrix(state).ToArray();
        int[] flattenedNextState = FlattenIntMatrix(nextState).ToArray();

        double maxOriginal = flattenedState.Max();
        double maxNext = flattenedNextState.Max();

        // If increases score, return increase of tile
        if (maxOriginal < maxNext)
            return maxNext - maxOriginal;

        // If less free space, return 0
        if (flattenedNextState.Count(i => i > 0) > flattenedState.Count(i => i > 0))
            return 0;

        // Otherwise, we had a combination--return 1
        return 1;
    }

    public bool IsTerminal(int[,] state)
    {
        IGame testGame = new Game(state);
        return testGame.Over;
    }

    public void PerformAction(Direction action)
    {
        Game.Action(action);
    }

    public void Restart()
    {
        Game.Restart();
    }

    /// <summary>
    /// Score is based solely on highest tile, for now
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    public double GetScore(int[,] state)
    {
        return FlattenIntMatrix(state).Max();
    }
    
    public Direction GetActionFromNodeNumber(int node)
    {
        if (node == 0)
            return Direction.UP;
        if (node == 1)
            return Direction.RIGHT;
        if (node == 2)
            return Direction.DOWN;
        return Direction.LEFT;
    }
    
    public int GetNodeNumberFromAction(Direction direction)
    {
        if (direction == Direction.UP)
            return 0;
        if (direction == Direction.RIGHT)
            return 1;
        if (direction == Direction.DOWN)
            return 2;
        return 3;
    }

    /// <summary>
    /// Turn an int matrix into an enumerable of ints
    /// </summary>
    /// <param name="matrix"></param>
    /// <returns></returns>
    private static IEnumerable<int> FlattenIntMatrix(int[,] matrix)
    {
        foreach (int i in matrix)
            yield return i;
    }


    // feature extractors

    private bool IncreasesHighestTile(int[,] state, Direction action)
    {
        IGame game = new Game(state);
        game.ActionNoAddTile(action);
        return FlattenIntMatrix(game.Board).Max() > FlattenIntMatrix(state).Max();
    }

    private static int TryGetDictionaryValueOrZero(Dictionary<int, int> dict, int key)
    {
        if (dict.TryGetValue(key, out int value))
            return value;
        return 0;
    }

    private bool HighestInCorner(int[,] state)
    {
        int max = FlattenIntMatrix(state).Max();
        return state[0, 0] == max || state[0, 3] == max || state[3, 0] == max || state[3, 3] == max;
    }

    private bool HighestInMiddle(int[,] state)
    {
        int max = FlattenIntMatrix(state).Max();
        return state[1, 1] == max || state[1, 2] == max || state[2, 1] == max || state[2, 2] == max;
    }

    private bool MaxTileMoves(int[] stateArr, int[] nextStateArr)
    {
        int maxOriginal = stateArr.Max();
        IEnumerable<int> maxIndexOriginal = stateArr.Where(t => t == maxOriginal);
        int maxNext = nextStateArr.Max();
        IEnumerable<int> maxIndexNext = nextStateArr.Where(t => t == maxNext);

        return !maxIndexOriginal.Intersect(maxIndexNext).Any();
    }
}