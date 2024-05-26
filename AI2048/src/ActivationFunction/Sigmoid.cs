namespace AI2048.Deep;

/// <summary>
/// Implements IActivationFunction as the sigmoid function: 1/(1+e^(-x))
/// </summary>
public class Sigmoid : IActivationFunction
{
    public double Activate(double input)
    {
        return 1 / (1 + Math.Exp(-input));
    }

    /// <summary>
    /// Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public double ActivationDerivative(double input)
    {
        double activated = Activate(input);
        return ActivationDerivativeUsingActivated(activated);
    }

    /// <summary>
    /// Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    /// </summary>
    /// <param name="activated"></param>
    /// <returns></returns>
    public double ActivationDerivativeUsingActivated(double activated)
    {
        return activated * (1 - activated);
    }
}