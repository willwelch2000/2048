namespace AI2048.Deep;

/// <summary>
/// Static class for reading and writing activation functions
/// </summary>
public static class ActivationFunctionReadWrite
{
    public static string Write(IActivationFunction activator) => 
        activator switch
        {
            LeakyReLU => "LeakyReLU",
            NoActivation => "NoActivation",
            ReLUWithSlopes => $"ReLUWithSlopes {((ReLUWithSlopes)activator).PositiveSlope} {((ReLUWithSlopes)activator).NegativeSlope}",
            Sigmoid => "Sigmoid",
            _ => throw new Exception("Unknown activation function type")
        };

    public static IActivationFunction Read(string description)
    {
        var words = description.Split();
        return words[0] switch
        {
            "LeakyReLU" => new LeakyReLU(),
            "NoActivation" => new NoActivation(),
            "ReLUWithSlopes" => new ReLUWithSlopes(double.Parse(words[1]), double.Parse(words[2])),
            "Sigmoid" => new Sigmoid(),
            _ => throw new Exception("Unknown activation function type"),
        };
    }
}