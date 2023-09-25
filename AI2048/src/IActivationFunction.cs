namespace AI2048.Deep;

/// <summary>
/// An object that can perform an activation function as part of a neural network.
/// This implements the strategy pattern for the network.
/// </summary>
public interface IActivationFunction
{
    public double Activate(double input);
    public double ActivationDerivative(double input);
}