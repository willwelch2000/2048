namespace AI2048.Deep;

/// <summary>
/// Implements IActivationFunction as the sigmoid function: 1/(1+e^(-x))
/// </summary>
public class LeakyReLU : IActivationFunction
{
    private const double subZeroSlope = 0.1;

    public double Activate(double x) =>
        x > 0 ? x : subZeroSlope * x;

    public double ActivationDerivative(double input, bool giveY) =>
        input > 0 ? 1 : subZeroSlope;
}