using MathNet.Numerics.LinearAlgebra;

namespace AI2048.Deep;


/// <summary>
/// This class contains all the information needed to transform 
/// one layer into the next layer: weights, biases, and activation function
/// output_layer = activation(weights * input_layer + biases)
/// </summary>
public class LayerTransform(Matrix<double> weights, Vector<double> biases)
{

    /// <summary>
    /// Each entry in the matrix is the weight from a starting node to an ending node
    /// indexed like w[end node, start node]
    /// </summary>
    public Matrix<double> Weights { get; internal set; } = weights;

    /// <summary>
    /// Each entry in the vector is the bias for a given node in the layer
    /// </summary>
    public Vector<double> Biases { get; internal set; } = biases;

    /// <summary>
    /// The activation function used for this transform
    /// If null, no activation is used
    /// </summary>
    public IActivationFunction? Activator { get; set; }

    /// <summary>
    /// Map transformation of input to output
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <returns></returns>
    public void TransformLayer(Vector<double> input, Vector<double> output) 
    {
        if (Activator is null)
            (Weights * input + Biases).Map(x => x, output);
        else
            (Weights * input + Biases).Map(Activator.Activate, output);
    }

    /// <summary>
    /// Take activator derivative of of given y value
    /// If no activator, the derivative is 1, because f(x)=x
    /// </summary>
    /// <param name="y">y value of point to take derivative of</param>
    /// <returns></returns>
    public double ActivationDerivative(double y)
    {
        if (Activator is not null)
            return Activator.ActivationDerivative(y);
        return 1;
    }

    public LayerTransform Clone() =>
        new(Weights, Biases)
        {
            Activator = Activator
        };
}