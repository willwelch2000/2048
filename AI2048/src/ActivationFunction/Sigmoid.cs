namespace AI2048.Deep;

/// <summary>
/// Implements IActivationFunction as the sigmoid function: 1/(1+e^(-x))
/// </summary>
public class Sigmoid : IActivationFunction
{
    public double Activate(double x) =>
        1 / (1 + Math.Exp(-x));

    /// <summary>
    /// Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = y * (1 - y)
    /// </summary>
    /// <param name="input">x or y</param>
    /// <param name="giveY">true if y is given</param>
    /// <returns></returns>
    public double ActivationDerivative(double input, bool giveY)
    {
        // If y is given, just return y * (1 - y)
        if (giveY)
            return input * (1 - input);
        // Otherwise, calculate y and return y * (1 - y)
        double y = Activate(input);
        return y * (1 - y);
    }
}