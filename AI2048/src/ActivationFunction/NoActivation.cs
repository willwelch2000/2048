namespace AI2048.Deep;

/// <summary>
/// Implements IActivationFunction as no activation: f(x) = x
/// </summary>
public class NoActivation : IActivationFunction
{
    public double Activate(double x) => x;

    public double ActivationDerivative(double y) => 1;
}