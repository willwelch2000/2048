using Game2048.Game;

namespace Game2048.Display;

public interface IDisplay
{
    /// <summary>
    /// Automatically display whenever a move is made
    /// </summary>
    /// <param name="game">game to follow</param>
    public void FollowGame(IGame game);

    /// <summary>
    /// Display current game state
    /// </summary>
    public void DisplayGame();
}