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
    public int NeuralNetInputLayerSize => 256;


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

    /// <summary>
    /// 16 features representing tile numbers, and 240 representing which (nonzero) tiles are the same
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    public Vector<double> GetNeuralNetFeatures(int[,] state)
    {
        int[] tiles = FlattenIntMatrix(state).ToArray();
        Vector<double> features = Vector<double>.Build.Dense(NeuralNetInputLayerSize);
        int i = 0;

        // Basic features--all tile values
        foreach (int val in state)
            features[i++] = (double) val / 10000;
        
        // Maps of which tiles are the same
        for (int j = 0; j < 16; j++)
            for (int k = 0; k < 16; k++)
                if (j != k)
                    features[i++] = tiles[j] == tiles[k] && tiles[j] != 0 ? 1 : 0;
        
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
    /// Testing different scoring methods
    /// </summary>
    /// <param name="state"></param>
    /// <param name="action"></param>
    /// <param name="nextState"></param>
    /// <returns></returns>
    public double GetReward(int[,] state, Direction action, int[,] nextState)
    {
        // return ValueCombined(FlattenIntMatrix(state).ToArray(), FlattenIntMatrix(nextState).ToArray());

        double maxOriginal = FlattenIntMatrix(state).Max();
        double maxNext = FlattenIntMatrix(nextState).Max();
        // If increases score, return increase of tile
        if (maxOriginal < maxNext)
            return maxNext - maxOriginal;
        // If less free space, return -1
        if (FlattenIntMatrix(nextState).Where(i => i > 0).Count() > FlattenIntMatrix(state).Where(i => i > 0).Count())
            return -1;
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

    /// <summary>
    /// Returns the value of squares that were combined.
    /// For example, if two 2s combine to form a 4, this returns 4
    /// </summary>
    /// <param name="stateArr">state as array</param>
    /// <param name="nextStateArr">nextState as array</param>
    /// <returns></returns>
    private double ValueCombined(int[] stateArr, int[] nextStateArr)
    {
        Dictionary<int, int> originalValues = new();
        Dictionary<int, int> nextValues = new();
        Dictionary<int, int> added = new();

        // count values in original state
        foreach(int i in stateArr)
        {
            if (!originalValues.ContainsKey(i))
                originalValues[i] = 0;
            originalValues[i] += 1;
        }

        // count values in next state
        foreach(int i in nextStateArr)
        {
            if (!nextValues.ContainsKey(i))
                nextValues[i] = 0;
            nextValues[i] += 1;
        }

        // for every number present, figure out the difference between next state and original
        int nextStateMax = nextValues.Keys.Where(x => x > 0).Max();
        int value = nextStateMax;
        while(value > 1)
        {
            if (!added.ContainsKey(value))
                added[value] = 0;
            if (!added.ContainsKey(value/2))
                added[value/2] = 0;
            added[value] += SafeGetDictionaryValue(nextValues, value) - SafeGetDictionaryValue(originalValues, value);
            added[value/2] += 2 * added[value];
            value /= 2;
        }

        // return sum of key * number of occurrences
        return added.Keys.Select(k => k * added[k]).Sum();
    }

    private int SafeGetDictionaryValue(Dictionary<int, int> dict, int key)
    {
        if (dict.ContainsKey(key))
            return dict[key];
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