namespace AI2048.Deep;

/// <summary>
/// Implements IActivationFunction as the leaky ReLU function
/// </summary>
public class LeakyReLU : IActivationFunction
{
    private const double subZeroSlope = 0.1;

    public double Activate(double x) =>
        x > 0 ? x : subZeroSlope * x;

    public double ActivationDerivative(double y) =>
        y > 0 ? 1 : subZeroSlope;
}