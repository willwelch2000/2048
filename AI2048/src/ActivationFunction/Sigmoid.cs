namespace AI2048.Deep;

/// <summary>
/// Implements IActivationFunction as the sigmoid function: f(x) = 1/(1+e^(-x))
/// </summary>
public class Sigmoid : IActivationFunction
{
    public double Activate(double x) =>
        1 / (1 + Math.Exp(-x));

    /// <summary>
    /// Derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = y * (1 - y)
    /// </summary>
    /// <param name="y">the y coordinate of the point to find the derivative of</param>
    /// <returns></returns>
    public double ActivationDerivative(double y) =>
        y * (1 - y);
}