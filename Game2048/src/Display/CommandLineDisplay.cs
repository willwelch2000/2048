using Game2048.Game;

namespace Game2048.Display;

public class CommandLineDisplay : IDisplay
{
    // // // fields

    IGame game;


    // // // constructors

    public 
    CommandLineDisplay(IGame game)
    {
        this.game = game;
        FollowGame(game);
    }


    // // // methods

    public void FollowGame(IGame game)
    {
        this.game = game;
        game.MoveLeft += DisplayGame;
        game.MoveRight += DisplayGame;
        game.MoveUp += DisplayGame;
        game.MoveDown += DisplayGame;
    }

    public void DisplayGame()
    {
        Console.WriteLine(Enumerable.Range(0, game.Dimension).Select(i => "+------").Append("+").Aggregate((current, next) => current + next));
        for (int i = 0; i < game.Dimension; i++) {
            Console.Write(Enumerable.Range(0, game.Dimension).Select(i => "|      ").Append("|\n|").Aggregate((current, next) => current + next));
            for (int j = 0; j < game.Dimension; j++)
                if (game[i, j] == 0)
                    Console.Write("      |");
                else
                    Console.Write($"{game[i, j], 6}|");
            Console.Write("\n" + Enumerable.Range(0, game.Dimension).Select(i => "|      ").Append("|").Aggregate((current, next) => current + next));
            Console.WriteLine("\n" + Enumerable.Range(0, game.Dimension).Select(i => "+------").Append("+").Aggregate((current, next) => current + next));
        }
    }

    private void DisplayGame(object? sender, EventArgs e)
    {
        DisplayGame();
    }
}