using MathNet.Numerics.LinearAlgebra;

namespace AI2048.Deep;


/// <summary>
/// This class contains all the information needed to transform 
/// one layer into the next layer: weights, biases, and activation function
/// output_layer = activation(weights * input_layer + biases)
/// </summary>
public class LayerTransform(Matrix<double> weights, Vector<double> biases, IActivationFunction activator)
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
    public IActivationFunction Activator { get; set; } = activator;

    /// <summary>
    /// Map transformation of input to output
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <returns></returns>
    public void TransformLayer(Vector<double> input, Vector<double> output) =>
        (Weights * input + Biases).Map(Activator.Activate, output);

    public LayerTransform Clone() => new(Weights, Biases, Activator);
}