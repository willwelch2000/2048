namespace AI2048.Deep;

/// <summary>
/// An object that can perform an activation function as part of a neural network.
/// This implements the strategy pattern for the network.
/// </summary>
public interface IActivationFunction
{
    public double Activate(double input);
    public double ActivationDerivative(double input);

    /// <summary>
    /// For some (i.e. sigmoid), it's more useful to input the already-activated number when finding the derivative
    /// This accepts activator(x) as the input and returns d_activator(x)/dx
    /// </summary>
    /// <param name="activated"></param>
    /// <returns></returns>
    public double ActivationDerivativeUsingActivated(double activated);
}