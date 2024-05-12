namespace AI2048.Deep;

public class Sigmoid : IActivationFunction
{
    public double Activate(double input)
    {
        return 1 / (1 + Math.Exp(-input));
    }
    public double ActivationDerivative(double input)
    {
        return input * (1 - input);
    }
}