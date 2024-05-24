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
    private Matrix<double>[] weights;
    private Vector<double>[] biases;
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
        weights[0] = Matrix<double>.Build.Random(numMiddleLayers > 0 ? numMiddleNodes : numOutputNodes, numInputNodes);
        weights[^1] = Matrix<double>.Build.Random(numOutputNodes, numMiddleLayers > 0 ? numMiddleNodes : numInputNodes);
        for (int i = 1; i < numMiddleLayers; i++)
            weights[i] = Matrix<double>.Build.Random(numMiddleNodes, numMiddleNodes);

        // Initialize biases
        biases = new Vector<double>[numMiddleLayers + 1];
        biases[^1] = Vector<double>.Build.Random(numOutputNodes);
        for (int i = 0; i < numMiddleLayers; i++)
            biases[i] = Vector<double>.Build.Random(numMiddleNodes);

        // Initialize activator
        this.activator = activator;

        // Initialize nodes
        nodes = new Vector<double>[numMiddleLayers + 2];
        nodes[0] = Vector<double>.Build.Dense(numInputNodes);
        nodes[^1] = Vector<double>.Build.Dense(numOutputNodes);
        for (int i = 1; i < numMiddleLayers + 1; i++)
            nodes[i] = Vector<double>.Build.Dense(numMiddleNodes);

        // Initialize cache dictionaries
        weightDerivativeCache = [];
        biasDerivativeCache = [];
        nodeDerivativeCache = [];
    }


    // // // properties

    /// <summary>
    /// Learning factor. How quickly weights change.
    /// </summary>
    public double Alpha { get; set; } = 10;

    public Vector<double>[] Nodes => nodes;
    public Matrix<double>[] Weights => weights;


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

        ResetDerivativeCaches();

        return nodes.Last();
    }

    public void SetWeight(int startLayer, int startNode, int endNode, double value)
    {
        weights[startLayer][endNode, startNode] = value;
        ResetDerivativeCaches();
    }

    public void SetBias(int startLayer, int endNode, double value)
    {
        biases[startLayer][endNode] = value;
        ResetDerivativeCaches();
    }

    /// <summary>
    /// Given a single set of input value and its desired output, adjust the weights accordingly
    /// </summary>
    /// <param name="input"></param>
    /// <param name="compare"></param>
    public void PerformGradientDescent(Vector<double> input, Vector<double> compare)
    {
        // Create new set of matrices for weights
        Matrix<double>[] newWeights = new Matrix<double>[weights.Length];
        for (int i = 0; i < weights.Length; i++)
            newWeights[i] = weights[i].Clone();
            
        // Create new set of vectors for biases
        Vector<double>[] newBiases = new Vector<double>[weights.Length];
        for (int i = 0; i < biases.Length; i++)
            newBiases[i] = biases[i].Clone();

        Vector<double> output = GetOutputValues(input);
        int outputLength = output.Count;

        // Iterate through all layers
        for (int layer = 0; layer < weights.Length; layer++)
        {
            Matrix<double> layerWeightMatrix = newWeights[layer];
            Vector<double> layerBiasVector = newBiases[layer];

            // Iterate through all end nodes of next layer
            for (int endNode = 0; endNode < layerWeightMatrix.RowCount; endNode++)
            {
                // Adjust all weights for the layer
                for (int wStartNode = 0; wStartNode < layerWeightMatrix.ColumnCount; wStartNode++)
                {
                    double wSum = 0;
                    for (int outputNode = 0; outputNode < outputLength; outputNode++)
                        wSum += (output[outputNode] - compare[outputNode]) * WeightDerivative(nodes.Length - 1, outputNode, layer, endNode, wStartNode);
                    layerWeightMatrix[endNode, wStartNode] -= 2 / ((double) outputLength) * Alpha*wSum;
                }

                // Adjust all biases for the layer
                double bSum = 0;
                for (int outputNode = 0; outputNode < outputLength; outputNode++)
                    bSum += 2 / ((double) outputLength) * (output[outputNode] - compare[outputNode]) * BiasDerivative(nodes.Length - 1, outputNode, layer, endNode);
                layerBiasVector[endNode] -= Alpha*bSum;
            }
        }

        weights = newWeights;
        biases = newBiases;
    }

    /// <summary>
    /// TODO
    /// </summary>
    /// <param name="input"></param>
    /// <param name="compare"></param>
    public void PerformGradientDescent(Vector<double>[] input, Vector<double>[] compare)
    {

    }

    /// <summary>
    /// Get derivative of a node with respect to a weight
    /// To calculate d(nodes[5][2])/d(w[1][2][3]): WeightDerivative(5, 2, 1, 2, 3)
    /// To work properly, the GetOutputValues() function must be called first
    /// </summary>
    /// <param name="layer">layer of the numerator node for the derivative. For example, with dy1/dw111, y is the layer</param>
    /// <param name="node">which node in the layer for the numerator for the derivative</param>
    /// <param name="wStartLayer">the start layer of the weight under examination</param>
    /// <param name="wEndNode">the end node of the weight under examination</param>
    /// <param name="wStartNode">the start node of the weight under examination</param>
    /// <returns></returns>
    public double WeightDerivative(int layer, int node, int wStartLayer, int wEndNode, int wStartNode)
    {
        // Already cached
        if (weightDerivativeCache.ContainsKey((layer, node, wStartLayer, wEndNode, wStartNode)))
            return weightDerivativeCache[(layer, node, wStartLayer, wEndNode, wStartNode)];

        // Simple case for derivative
        if (layer == wStartLayer + 1)
        {
            double derivative;
            if (wEndNode == node)
                derivative = activator.ActivationDerivative(nodes[layer][node]) * nodes[layer - 1][wStartNode];
            else
                // This bias has nothing to do with that node
                derivative = 0;
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
    /// To work properly, the GetOutputValues() function must be called first
    /// </summary>
    /// <param name="layer">layer of the numerator node for the derivative</param>
    /// <param name="node">which node in the target layer for the derivative</param>
    /// <param name="bStartLayer">the starting layer (layer before the one that it's used to calculate) of the bias that is the denominator</param>
    /// <param name="bEndNode">the node that this bias is involved in calculating</param>
    /// <returns></returns>
    public double BiasDerivative(int layer, int node, int bStartLayer, int bEndNode)
    {
        // Already cached
        if (biasDerivativeCache.ContainsKey((layer, node, bStartLayer, bEndNode)))
            return biasDerivativeCache[(layer, node, bStartLayer, bEndNode)];

        // Simple case for derivative
        if (layer == bStartLayer + 1)
        {
            double derivative;

            if (bEndNode == node)
                derivative = activator.ActivationDerivative(nodes[layer][bEndNode]);
            else
                // This bias has nothing to do with that node
                derivative = 0;
            biasDerivativeCache[(layer, node, bStartLayer, bEndNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(db__) = Sum{(dhi_)/(dhi-1_) * (dhi-1)/(db__)}
        if (bStartLayer + 1 < layer)
        {
            int prevLayerLength = nodes[layer - 1].Count;
            double derivative = Enumerable.Range(0, prevLayerLength).Select(i => NodeDerivative(layer, node, layer - 1, i) * BiasDerivative(layer - 1, i, bStartLayer, bEndNode)).Sum();
            biasDerivativeCache[(layer, node, bStartLayer, bEndNode)] = derivative;
            return derivative;
        }

        return 0;
    }

    /// <summary>
    /// Get derivative of node with respect to another node
    /// To work properly, the GetOutputValues() function must be called first
    /// </summary>
    /// <param name="endLayer">layer of the numerator node for the derivative</param>
    /// <param name="endNode">which node in the target layer for the derivative</param>
    /// <param name="startLayer">layer of the denominator node</param>
    /// <param name="startNode">which node in the layer for the denominator node</param>
    /// <returns></returns>
    public double NodeDerivative(int endLayer, int endNode, int startLayer, int startNode)
    {
        // Already cached
        if (nodeDerivativeCache.ContainsKey((endLayer, endNode, startLayer, startNode)))
            return nodeDerivativeCache[(endLayer, endNode, startLayer, startNode)];

        // Simple case for derivative
        if (endLayer == startLayer + 1)
        {
            double derivative = activator.ActivationDerivative(nodes[endLayer][endNode]) * weights[startLayer][endNode, startNode];
            nodeDerivativeCache[(endLayer, endNode, startLayer, startNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(db__) = Sum{(dhi_)/(dhi-1_) * (dhi-1)/(db__)}
        if (startLayer + 1 < endLayer)
        {
            int prevLayerLength = nodes[endLayer - 1].Count;
            double derivative = Enumerable.Range(0, prevLayerLength).Select(i => NodeDerivative(endLayer, endNode, endLayer - 1, i) * NodeDerivative(endLayer - 1, i, startLayer, startNode)).Sum();
            nodeDerivativeCache[(endLayer, endNode, startLayer, startNode)] = derivative;
            return derivative;
        }

        return 0;
    }

    public void ResetDerivativeCaches()
    {
        weightDerivativeCache = [];
        biasDerivativeCache = [];
        nodeDerivativeCache = [];
    }

    public NeuralNet Clone()
    {
        NeuralNet clone = new(nodes[0].Count, nodes[1].Count, nodes[^1].Count, weights.Length - 1);
        for (int i = 0; i < weights.Length; i++)
        {
            clone.weights[i] = weights[i].Clone();
            clone.biases[i] = biases[i].Clone();
        }
        return clone;
    }
}