using MathNet.Numerics.LinearAlgebra;

namespace AI2048.Deep;

public class NeuralNet
{
    // // // fields
    /// <summary>
    /// Array of matrices
    /// Each entry in a matrix is the weight from a starting node to an ending node
    /// indexed like w[end node, start node]
    /// There is a matrix for each layer-to-layer transformation
    /// matrix * (layer as column vector) = next layer
    /// </summary>
    private readonly Matrix<double>[] weights;
    private readonly Vector<double>[] biases;
    private readonly IActivationFunction activator;
    private readonly Vector<double>[] nodes;
    private Dictionary<(int layer, int node, int wStartLayer, int wEndNode, int wStartNode), double> weightDerivativeCache;
    private Dictionary<(int layer, int node, int bStartLayer, int bEndNode), double> biasDerivativeCache;
    private Dictionary<(int endLayer, int endNode, int startLayer, int startNode), double> nodeDerivativeCache;


    // // // constructors

    public NeuralNet(int numInputNodes, int numMiddleNodes, int numOutputNodes, int numMiddleLayers) : this(numInputNodes, numMiddleNodes, numOutputNodes, numMiddleLayers, new Sigmoid()) {}

    public NeuralNet(int numInputNodes, int numMiddleNodes, int numOutputNodes, int numMiddleLayers, IActivationFunction activator)
    {
        // Initialize weights
        weights = new Matrix<double>[numMiddleLayers + 1];
        weights[0] = Matrix<double>.Build.Dense(numMiddleLayers > 0 ? numMiddleNodes : numOutputNodes, numInputNodes);
        weights[^1] = Matrix<double>.Build.Dense(numOutputNodes, numMiddleLayers > 0 ? numMiddleNodes : numInputNodes);
        for (int i = 1; i < numMiddleLayers + 1; i++)
            weights[i] = Matrix<double>.Build.Dense(numMiddleNodes, numMiddleNodes);

        // Initialize biases
        biases = new Vector<double>[numMiddleLayers + 1];
        biases[^1] = Vector<double>.Build.Dense(numOutputNodes);
        for (int i = 0; i < numMiddleLayers + 1; i++)
            biases[i] = Vector<double>.Build.Dense(numMiddleNodes);

        // Initialize activator
        this.activator = activator;

        // Initialize nodes
        nodes = new Vector<double>[numMiddleLayers + 2];
        nodes[0] = Vector<double>.Build.Dense(numInputNodes);
        nodes[^1] = Vector<double>.Build.Dense(numOutputNodes);
        for (int i = 1; i < numMiddleLayers + 1; i++)
            nodes[i] = Vector<double>.Build.Dense(numMiddleNodes);

        // Initialize cache dictionaries
        weightDerivativeCache = new();
        biasDerivativeCache = new();
        nodeDerivativeCache = new();
    }


    // // // methods

    /// <summary>
    /// Given the input nodes as an array, get the resulting output node values
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public Vector<double> GetOutputValues(Vector<double> input)
    {
        nodes[0] = input.Clone();
        for (int layer = 0; layer < weights.Length; layer++)
        {
            Matrix<double> layerWeights = weights[layer];
            Vector<double> layerBiases = biases[layer];

            // Perform matrix operation to get from currentLayer to nextLayer
            (layerWeights * nodes[layer] + layerBiases).Map(activator.Activate, nodes[layer + 1]);
        }

        return nodes.Last();
    }

    public void SetWeight(int startLayer, int startNode, int nextNode, double value)
    {
        weights[startLayer][nextNode, startNode] = value;
    }

    public void SetBias(int startLayer, int endNode, double value)
    {
        biases[startLayer][endNode] = value;
    }

    public void PerformGradientDescent(Vector<double> input, Vector<double> compare)
    {

    }

    /// <summary>
    /// Get derivative of a node with respect to a weight
    /// To calculate d(nodes[5][2])/d(w[1][2][3]): WeightDerivative(5, 2, 1, 2, 3)
    /// </summary>
    /// <param name="layer">layer of the numerator for the derivative. For example, with dy1/dw111, y is the layer</param>
    /// <param name="node">which node in the layer for the numerator for the derivative</param>
    /// <param name="wStartLayer">the start layer of the weight under examination</param>
    /// <param name="wEndNode">the end node of the weight under examination</param>
    /// <param name="wStartNode">the start node of the weight under examination</param>
    /// <returns></returns>
    private double WeightDerivative(int layer, int node, int wStartLayer, int wEndNode, int wStartNode)
    {
        // Already cached
        if (weightDerivativeCache.ContainsKey((layer, node, wStartLayer, wEndNode, wStartNode)))
            return weightDerivativeCache[(layer, node, wStartLayer, wEndNode, wStartNode)];

        // Simple case for derivative
        if (layer == wStartLayer + 1)
        {
            double derivative = activator.ActivationDerivative(nodes[layer][wEndNode]) * nodes[layer - 1][wStartNode];
            weightDerivativeCache[(layer, node, wStartLayer, wEndNode, wStartNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(dw___) = Sum{(dhi_)/(dhi-1_) * (dhi-1)/(dw___)}
        if (wStartLayer + 1 < layer)
        {
            int prevLayerLength = nodes[layer - 1].Count;
            double derivative = Enumerable.Range(0, prevLayerLength).Select(i => NodeDerivative(layer, node, layer - 1, i) * WeightDerivative(layer - 1, i, wStartLayer, wEndNode, wStartNode)).Sum();
            weightDerivativeCache[(layer, node, wStartLayer, wEndNode, wStartNode)] = derivative;
            return derivative;
        }

        return 0;
    }

    /// <summary>
    /// Get derivative of node with respect to a bias
    /// </summary>
    /// <param name="layer"></param>
    /// <param name="node"></param>
    /// <param name="bStartLayer"></param>
    /// <param name="bEndNode"></param>
    /// <returns></returns>
    private double BiasDerivative(int layer, int node, int bStartLayer, int bEndNode)
    {
        // Already cached
        if (biasDerivativeCache.ContainsKey((layer, node, bStartLayer, bEndNode)))
            return biasDerivativeCache[(layer, node, bStartLayer, bEndNode)];

        return 0;
    }

    /// <summary>
    /// Get derivative of node with respect to another node
    /// </summary>
    /// <param name="endLayer"></param>
    /// <param name="endNode"></param>
    /// <param name="startLayer"></param>
    /// <param name="startNode"></param>
    /// <returns></returns>
    private double NodeDerivative(int endLayer, int endNode, int startLayer, int startNode)
    {
        // Already cached
        if (biasDerivativeCache.ContainsKey((endLayer, endNode, startLayer, startNode)))
            return biasDerivativeCache[(endLayer, endNode, startLayer, startNode)];

        return 0;
    }
}