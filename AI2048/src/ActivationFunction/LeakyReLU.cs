namespace AI2048.Deep;

/// <summary>
/// Implements IActivationFunction as the leaky ReLU function: f(x) = x > 0 ? x : 0.1x
/// </summary>
public class LeakyReLU : IActivationFunction
{
    private const double subZeroSlope = 0.01;

    public double Activate(double x) =>
        x > 0 ? x : subZeroSlope * x;

    public double ActivationDerivative(double y) =>
        y > 0 ? 1 : subZeroSlope;
}