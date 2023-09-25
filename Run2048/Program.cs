using Game2048.Display;
using Game2048.Game;

namespace Run2048;

public class Program
{
    public static void Main()
    {
        IGame game= new Game();
        IDisplay display = new CommandLineDisplay(game);
        display.DisplayGame();

        while (!game.Over)
        {
            Console.Write("Enter a move: ");
            string input = Console.ReadLine() ?? "";
            if (input.Equals("close"))
                break;
            Direction move = ParseDirection(input);
            game.Action(move);
        }
    }

    private static Direction ParseDirection(string input)
    {
        switch (input.ToLower())
        {
            case "left":
                return Direction.LEFT;
            case "right":
                return Direction.RIGHT;
            case "up":
                return Direction.UP;
            case "down":
                return Direction.DOWN;
            default:
                return Direction.LEFT;
        }
    }
}