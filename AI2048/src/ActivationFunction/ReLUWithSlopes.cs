namespace AI2048.Deep;

/// <summary>
/// Implements IActivationFunction as a modified ReLU function, with defined positive and negative slopes
/// </summary>
public class ReLUWithSlopes(double positiveSlope, double negativeSlope) : IActivationFunction
{
    public double Activate(double x) =>
        x > 0 ? positiveSlope * x : negativeSlope * x;

    public double ActivationDerivative(double y) =>
        y > 0 ? positiveSlope : negativeSlope;
}