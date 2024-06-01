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
    /// Find the derivative of the activation function at the given y value
    /// We know the y value, so that is what must be passed in
    /// </summary>
    /// <param name="y">the y coordinate of the point to find the derivative of</param>
    /// <returns></returns>
    public double ActivationDerivative(double y);
}