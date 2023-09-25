namespace Game2048.Game;

public interface IGame
{
    // // // events

    public event EventHandler? MoveLeft;
    public event EventHandler? MoveRight;
    public event EventHandler? MoveUp;
    public event EventHandler? MoveDown;


    // // // indexers

    /// <summary>
    /// Two dimensional indexer accesses board square
    /// </summary>
    public int this[int row, int column] { get; }


    // // // properties

    /// <summary>
    /// true if board is filled, false otherwise
    /// </summary>
    public bool Filled { get; }
    
    /// <summary>
    /// Iterates through all coordinates of the board
    /// </summary>
    public IEnumerable<(int row, int column)> Coordinates { get; }

    /// <summary>
    /// Accessor for private value dimension
    /// </summary>
    public int Dimension { get; }

    /// <summary>
    /// true if the game is over, false otherwise
    /// </summary>
    public bool Over { get; }

    /// <summary>
    /// The number of empty spaces on the board
    /// </summary>
    public int EmptySpaces { get; }

    /// <summary>
    /// The value of the highest tile on the board
    /// </summary>
    public int HighestNumber { get; }

    /// <summary>
    /// A matrix representing the board
    /// </summary>
    public int[,] Board { get; }


    // // // methods

    /// <summary>
    /// Adds a new tile to the board. 
    /// </summary>
    public void AddTile();

    /// <summary>
    /// Tests if a given space is empty or not
    /// </summary>
    /// <param name="row">The row of the space</param>
    /// <param name="column">The column of the space</param>
    /// <returns>true if empty, false if filled</returns>
    public bool Empty(int row, int column);

    /// <summary>
    /// Tests if the given move is possible
    /// </summary>
    /// <param name="direction">Which direction to test</param>
    /// <returns>true if possible, false otherwise</returns>
    public bool IsPossibleMove(Direction direction);

    /// <summary>
    /// Find all coordinates of tiles whose value is the given number--each coordinate is (row, column)
    /// </summary>
    /// <param name="number"></param>
    /// <returns>Collection of coordinates, where each coordinate is (int, int)</returns>
    public IEnumerable<(int row, int column)> FindCoordinates(int number);

    /// <summary>
    /// Perform the given action
    /// </summary>
    /// <param name="direction">Direction enum representing which direction to move</param>
    public void Action(Direction direction);

    /// <summary>
    /// Perform the given action without adding a tile.
    /// Used for testing purposes
    /// </summary>
    /// <param name="direction">Direction enum representing which direction to move</param>
    public void ActionNoAddTile(Direction direction);

    /// <summary>
    /// Allow an outside class to link an event handler to a specific move
    /// </summary>
    /// <param name="eventHandler"></param>
    /// <param name="direction">which direction to track</param>
    public void Subscribe(EventHandler eventHandler, Direction direction);
}