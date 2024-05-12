namespace Game2048.Game;

/// <summary>
/// Represents a 2048 Game.
/// Allows outside code to simulate gameplay
/// </summary>
public class Game : IGame
{
    // // // fields
    
    private const int dimension = 4;
    private const double percentageTwos = 0.9;
    private readonly Random random = new();

    /// <summary>
    /// Matrix representing the board of the 2048 Game.
    /// board[0, 0] is top left tile, board[3, 3] is the bottom right.
    /// 0 represents empty tile.
    /// </summary>
    private readonly int[,] board = new int[dimension, dimension];


    // // // events

    // All moves cause an event, so that display classes can track
    public event EventHandler? MoveLeft;
    public event EventHandler? MoveRight;
    public event EventHandler? MoveUp;
    public event EventHandler? MoveDown;


    // // // constructors

    /// <summary>
    /// Default constructor--starts empty and adds two randomized tiles. 
    /// </summary>
    public Game()
    {
        AddTile();
        AddTile();
    }

    /// <summary>
    /// Constructor that generates a copy of another game. 
    /// </summary>
    public Game(IGame other)
    {
        foreach ((int row, int column) in Coordinates)
            board[row, column] = other[row, column];
    }

    /// <summary>
    /// Constructor given a starting board as a 4x4 int matrix
    /// </summary>
    /// <param name="board">a 4x4 int matrix representing the board</param>
    public Game(int[,] board)
    {
        if (board.GetLength(0) != dimension || board.GetLength(1) != dimension)
            throw new ArgumentException("board must be a 4x4 matrix.");
        foreach ((int row, int column) in Coordinates)
            this.board[row, column] = board[row, column];
    }


    // // // indexers

    /// <summary>
    /// Two dimensional 0-based indexer accesses board square
    /// </summary>
    public int this[int row, int column]
    {
        get {
            TestValidity(row, column);
            if (row < 0 || row >= dimension)
                return 0;
            return board[row, column];
        }
    }


    // // // properties

    public bool Filled =>
        !Coordinates.Where(c => board[c.row, c.column] == 0).Any();

    /// <summary>
    /// Iterates through all coordinates of the board
    /// </summary>
    public IEnumerable<(int row, int column)> Coordinates
    {
        get {
            for (int row = 0; row < dimension; row++)
                for (int column = 0; column < dimension; column++)
                    yield return new (row, column);
        }
    }

    /// <summary>
    /// Accessor for private value dimension
    /// </summary>
    public int Dimension => dimension;

    /// <summary>
    /// Tests if the game is over
    /// </summary>
    public bool Over
    {
        get
        {
            Game copy = new(this);
            foreach (Direction direction in Enum.GetValues(typeof(Direction)))
            {
                copy.Action(direction);
                if (!Equals(copy))
                    return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Number of empty spaces on the board
    /// </summary>
    public int EmptySpaces =>
        Coordinates.Select(coord => board[coord.row, coord.column]).Where(val => val == 0).Count();

    /// <summary>
    /// Value of highest tile on the board
    /// </summary>
    public int HighestNumber =>
        Coordinates.Select(coord => board[coord.row, coord.column]).Max();

    public int[,] Board =>
        GetBoardCopy();


    // // // methods

    /// <summary>
    /// Adds a new tile to the board. The location is chosen randomly
    /// from the set of existing empty squares. 90% of the time, a 2 is
    /// chosen as the new value. A 4 is chosen the remaining 10%.
    /// </summary>
    /// <exception cref="Exception">Throws exception if the board is filled</exception>
    public void AddTile()
    {
        if (Filled)
            throw new Exception("Cannot add tile. Board is filled.");

        int locationChoice = (int) random.NextInt64(dimension*dimension);
        while (board[locationChoice / 4, locationChoice % 4] != 0)
            locationChoice = (int) random.NextInt64(dimension*dimension);

        int numberChoice = random.NextDouble() < percentageTwos ? 2 : 4;

        board[locationChoice / 4, locationChoice % 4] = numberChoice;
    }

    /// <summary>
    /// Tests if a given space is empty or not
    /// </summary>
    /// <param name="row">The row of the space</param>
    /// <param name="column">The column of the space</param>
    /// <returns>true if empty, false if filled</returns>
    public bool Empty(int row, int column)
    {
        TestValidity(row, column);
        return this[row, column] == 0;
    }

    /// <summary>
    /// Tests if this game is the same as another
    /// </summary>
    /// <param name="other">The game to compare to</param>
    /// <returns>true if equal, false otherwise</returns>
    public override bool Equals(object? obj)
    {
        if (obj is Game other)
            return Coordinates.All(coord => this[coord.row, coord.column] == other[coord.row, coord.column]);
        return false;
    }

    public override int GetHashCode()
    {
        return base.GetHashCode();
    }

    public static bool operator == (Game game1, Game game2)
    {
        return game1.Equals(game2);
    }

    public static bool operator != (Game game1, Game game2)
    {
        return !game1.Equals(game2);
    }

    /// <summary>
    /// Tests if the given move is possible
    /// </summary>
    /// <param name="direction">Which direction to test</param>
    /// <returns>true if possible, false otherwise</returns>
    public bool IsPossibleMove(Direction direction)
    {
        Game copy = new(this);
        copy.Action(direction);
        return !Equals(copy);
    }

    public IEnumerable<(int row, int column)> FindCoordinates(int number) =>
        Coordinates.Where(coord => board[coord.row, coord.column] == number);

    /// <summary>
    /// Perform the given action
    /// </summary>
    /// <param name="direction"></param>
    public void Action(Direction direction)
    {
        Game copy = new(this);
        if (direction == Direction.LEFT)
        {
            LeftBase();

            if (!Equals(copy))
                AddTile();
            MoveLeft?.Invoke(this, new EventArgs());
        }
        if (direction == Direction.RIGHT)
        {
            RotateCW(2);
            LeftBase();
            RotateCW(2);

            if (!Equals(copy))
                AddTile();
            MoveRight?.Invoke(this, new EventArgs());
        }
        if (direction == Direction.UP)
        {
            RotateCW(3);
            LeftBase();
            RotateCW(1);

            if (!Equals(copy))
                AddTile();
            MoveUp?.Invoke(this, new EventArgs());
        }
        if (direction == Direction.DOWN)
        {
            RotateCW(1);
            LeftBase();
            RotateCW(3);

            if (!Equals(copy))
                AddTile();
            MoveDown?.Invoke(this, new EventArgs());
        }
    }

    /// <summary>
    /// Do the given action without adding a tile.
    /// Only used for testing purposes
    /// </summary>
    /// <param name="direction"></param>
    public void ActionNoAddTile(Direction direction)
    {
        if (direction == Direction.LEFT)
        {
            LeftBase();
        }
        if (direction == Direction.RIGHT)
        {
            RotateCW(2);
            LeftBase();
            RotateCW(2);
        }
        if (direction == Direction.UP)
        {
            RotateCW(3);
            LeftBase();
            RotateCW(1);
        }
        if (direction == Direction.DOWN)
        {
            RotateCW(1);
            LeftBase();
            RotateCW(3);
        }
    }

    /// <summary>
    /// A move to the left
    /// This method is the base for moving in any direction
    /// </summary>
    private void LeftBase()
    {
        bool[,] combined = new bool[dimension, dimension];
        foreach ((int row, int column) in Coordinates)
            combined[row, column] = false;

        // Column 0

        // Column 1
        for (int row = 0; row < dimension; row++)
        {
            if (board[row, 1] == 0)
                continue;
            if (board[row, 0] == 0)
            {
                board[row, 0] = board[row, 1];
                board[row, 1] = 0;
            }
            else if (board[row, 0] == board[row, 1] && !combined[row, 0])
            {
                board[row, 0] *= 2;
                board[row, 1] = 0;
                combined[row, 0] = true;
            }
        }

        // Column 2
        for (int row = 0; row < dimension; row++)
        {
            if (board[row, 2] == 0)
                continue;
            if (board[row, 1] == 0)
            {
                board[row, 1] = board[row, 2];
                board[row, 2] = 0;
                if (board[row, 0] == 0)
                {
                    board[row, 0] = board[row, 1];
                    board[row, 1] = 0;
                }
                else if (board[row, 0] == board[row, 1] && !combined[row, 0])
                {
                    board[row, 0] *= 2;
                    board[row, 1] = 0;
                    combined[row, 0] = true;
                }
            }
            else if (board[row, 1] == board[row, 2] && !combined[row, 1])
            {
                board[row, 1] *= 2;
                board[row, 2] = 0;
                combined[row, 1] = true;
            }
        }

        // Column 3
        for (int row = 0; row < dimension; row++)
        {
            if (board[row, 3] == 0)
                continue;
            if (board[row, 2] == 0)
            {
                board[row, 2] = board[row, 3];
                board[row, 3] = 0;
                if (board[row, 1] == 0)
                {
                    board[row, 1] = board[row, 2];
                    board[row, 2] = 0;
                    if (board[row, 0] == 0)
                    {
                        board[row, 0] = board[row, 1];
                        board[row, 1] = 0;
                    }
                    else if (board[row, 0] == board[row, 1] && combined[row, 0])
                    {
                        board[row, 0] *= 2;
                        board[row, 1] = 0;
                        combined[row, 0] = true;
                    }
                }
                else if (board[row, 1] == board[row, 2] && !combined[row, 1])
                {
                    board[row, 1] *= 2;
                    board[row, 2] = 0;
                    combined[row, 1] = true;
                }
            }
            else if (board[row, 2] == board[row, 3] && !combined[row, 2])
            {
                board[row, 2] *= 2;
                board[row, 3] = 0;
                combined[row, 2] = true;
            }
        }
    }

    public void Subscribe(EventHandler eventHandler, Direction direction)
    {
        switch (direction)
        {
            case Direction.LEFT:
                MoveLeft += eventHandler;
                break;
            case Direction.RIGHT:
                MoveRight += eventHandler;
                break;
            case Direction.UP:
                MoveUp += eventHandler;
                break;
            case Direction.DOWN:
                MoveDown += eventHandler;
                break;
        }
    }

    /// <summary>
    /// Rotate the board clockwise a given number of times. Used as helper function
    /// </summary>
    /// <param name="numRotations"></param>
    private void RotateCW(int numRotations)
    {
        for (int _ = 0; _ < numRotations; _++)
        {
            int[,] boardCopy = GetBoardCopy();
            foreach ((int row, int column) in Coordinates)
                board[row, column] = boardCopy[dimension - 1 - column, row];
        }
    }

    private int[,] GetBoardCopy()
    {
        int[,] boardCopy = new int[dimension, dimension];
        foreach ((int row, int column) in Coordinates)
            boardCopy[row, column] = board[row, column];
        return boardCopy;
    }

    private static void TestValidity(int row, int column)
    {
        if (row >= dimension || column >= dimension)
            throw new ArgumentException($"row and column must be below {dimension}.");
        if (row < 0 || column < 0)
            throw new ArgumentException("row and column must be positive.");
    }
}
