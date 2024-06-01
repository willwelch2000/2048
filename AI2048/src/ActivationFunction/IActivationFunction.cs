namespace AI2048.Deep;

/// <summary>
/// An object that can perform an activation function as part of a neural network.
/// This implements the strategy pattern for the network.
/// </summary>
public interface IActivationFunction
{
    /// <summary>
    /// Perform the activation function on x
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public double Activate(double x);

    /// <summary>
    /// Find the derivative of the activation function at the given point
    /// For some (i.e. sigmoid), it's more useful to input the already-activated number (y) when finding the derivative
    /// So this accepts x or activator(x) as the input and returns d_activator(x)/dx
    /// </summary>
    /// <param name="input">the x or y coordinate of the point to find the derivative of</param>
    /// <param name="giveY">true if the y value was given</param>
    /// <returns></returns>
    public double ActivationDerivative(double input, bool giveY);
}