using MathNet.Numerics.LinearAlgebra;

namespace AI2048.Deep;

/// <summary>
/// Implements a nerual network
/// Calculation is done like this:
///     nodes[i+1] = activator(weights[i] * nodes[i] + biases[i])
/// </summary>
public class NeuralNet
{
    // // // fields
    
    private readonly LayerTransform[] layerTransforms;

    /// <summary>
    /// All nodes in the network
    /// </summary>
    private readonly Vector<double>[] nodes;

    // caches for derivative calculations

    private Dictionary<(int layer, int node, int wStartLayer, int wEndNode, int wStartNode), double> weightDerivativeCache;
    private Dictionary<(int layer, int node, int bStartLayer, int bEndNode), double> biasDerivativeCache;
    private Dictionary<(int endLayer, int endNode, int startLayer, int startNode), double> nodeDerivativeCache;


    // // // constructors

    public NeuralNet(int numInputNodes, int numMiddleNodes, int numOutputNodes, int numMiddleLayers) : this(numInputNodes, numMiddleNodes, numOutputNodes, numMiddleLayers, new Sigmoid()) {}


    /// <summary>
    /// Main constructor
    /// </summary>
    /// <param name="numInputNodes">Number of nodes for input layer</param>
    /// <param name="numMiddleNodes">Number of nodes for middle layers</param>
    /// <param name="numOutputNodes">Number of nodes for output layer</param>
    /// <param name="numMiddleLayers">Number of middle layers</param>
    /// <param name="activator"></param>
    public NeuralNet(int numInputNodes, int numMiddleNodes, int numOutputNodes, int numMiddleLayers, IActivationFunction primaryActivator)
    {
        // Initialize layer transforms
        layerTransforms = new LayerTransform[numMiddleLayers + 1];

        // First transform
        layerTransforms[0] = new(
            Matrix<double>.Build.Random(numMiddleLayers > 0 ? numMiddleNodes : numOutputNodes, numInputNodes),
            Vector<double>.Build.Random(numMiddleNodes),
            primaryActivator);

        // Last transform
        layerTransforms[^1] = new(
            Matrix<double>.Build.Random(numOutputNodes, numMiddleLayers > 0 ? numMiddleNodes : numInputNodes),
            Vector<double>.Build.Random(numOutputNodes),
            primaryActivator);

        // All other transforms
        for (int i = 1; i < numMiddleLayers; i++)
            layerTransforms[i] = new(
                Matrix<double>.Build.Random(numMiddleNodes, numMiddleNodes),
                Vector<double>.Build.Random(numMiddleNodes),
                primaryActivator);

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
    /// Access to layer transforms used to propagate input to output
    /// These are the actual layer transform objects, not copies, but the array itself (and its length) can't be changed from this
    /// </summary>
    public IEnumerable<LayerTransform> LayerTransforms => layerTransforms.AsEnumerable();

    /// <summary>
    /// Learning factor. How quickly weights change.
    /// </summary>
    public double Alpha { get; set; } = 0.01;

    /// <summary>
    /// Accessor for node values in the network
    /// Not actual node objects--copies
    /// </summary>
    public IEnumerable<Vector<double>> Nodes => nodes.Select(n => n.Clone());

    public int NumInputNodes => nodes[0].Count;
    public int NumMiddleNodes => nodes[^2].Count;
    public int NumOutputNodes => nodes[^1].Count;
    public int NumMiddleLayers => nodes.Length - 2;


    // // // methods

    /// <summary>
    /// Given the input nodes as an array, get the resulting output node values
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public Vector<double> GetOutputValues(Vector<double> input)
    {
        nodes[0] = input.Clone();
        for (int i = 0; i < layerTransforms.Length; i++)
            layerTransforms[i].TransformLayer(nodes[i], nodes[i + 1]);

        return nodes.Last().Clone();
    }

    public void SetWeight(int startLayer, int startNode, int endNode, double value) =>
        layerTransforms[startLayer].Weights[endNode, startNode] = value;

    public void SetBias(int startLayer, int endNode, double value) =>
        layerTransforms[startLayer].Biases[endNode] = value;

    public void SetActivator(int startLayer, IActivationFunction activator) =>
        layerTransforms[startLayer].Activator = activator;

    /// <summary>
    /// Given a single set of input value and its desired output, adjust the weights accordingly
    /// TODO: optimize
    /// </summary>
    /// <param name="input">input layer</param>
    /// <param name="compare">desired output</param>
    public void PerformGradientDescent(Vector<double> input, Vector<double> compare)
    {
        ResetDerivativeCaches();

        Vector<double> output = GetOutputValues(input);
        int outputLength = output.Count;

        // Iterate through all layers
        for (int startLayer = 0; startLayer < layerTransforms.Length; startLayer++)
        {
            Matrix<double> layerWeightMatrix = layerTransforms[startLayer].Weights.Clone();
            Vector<double> layerBiasVector = layerTransforms[startLayer].Biases.Clone();

            // Iterate through all end nodes of next layer
            for (int endNode = 0; endNode < layerWeightMatrix.RowCount; endNode++)
            {
                // Adjust all weights for the layer: w = w - Alpha*d/dw(Loss)
                // Loss = SUM((output - compare)^2)/outputlength
                // d/dw(Loss) = SUM(2(output - compare)*d/dw(output))/outputlength
                // Weight adjustment simplifies to: w = w - 2/outputlength * Alpha * SUM((output - compare)*d/dw(output))
                for (int wStartNode = 0; wStartNode < layerWeightMatrix.ColumnCount; wStartNode++)
                {
                    double wSum = 0;
                    for (int outputNode = 0; outputNode < outputLength; outputNode++)
                        wSum += (output[outputNode] - compare[outputNode]) * WeightDerivative(nodes.Length - 1, outputNode, startLayer, endNode, wStartNode);
                    layerWeightMatrix[endNode, wStartNode] -= 2 / ((double) outputLength) * Alpha*wSum;
                }

                // Adjust all biases for the layer--see above for explanation
                double bSum = 0;
                for (int outputNode = 0; outputNode < outputLength; outputNode++)
                    bSum += (output[outputNode] - compare[outputNode]) * BiasDerivative(nodes.Length - 1, outputNode, startLayer, endNode);
                layerBiasVector[endNode] -= 2 / ((double) outputLength) * Alpha*bSum;
            }

            layerTransforms[startLayer].Weights = layerWeightMatrix;
            layerTransforms[startLayer].Biases = layerBiasVector;
        }
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
        if (weightDerivativeCache.TryGetValue((layer, node, wStartLayer, wEndNode, wStartNode), out double value))
            return value;

        // Simple case for derivative
        if (layer == wStartLayer + 1)
        {
            double derivative;
            if (wEndNode == node)
                // endnode = activator(w*startnode + ...), so d_endnode/dw = d_activator/d_activator_input * d_activator_input/dw = d_activator/d_activator_input * startnode
                // Instead of actually giving the sigmoid input to the activator derivative function, we give the
                // already-activated value, just because the calculation is simpler (sigmoid*(1-sigmoid)), and it's what we know: the node value
                derivative = layerTransforms[wStartLayer].Activator.ActivationDerivative(nodes[layer][node]) * nodes[layer - 1][wStartNode];
            else
                // This weight has nothing to do with that node
                derivative = 0;
            weightDerivativeCache[(layer, node, wStartLayer, wEndNode, wStartNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(dw___) = Sum{(dhi_)/(dhi-1_) * (dhi-1_)/(dw___)}
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
                // endnode = activator(... + bias), so d_endnode/d_bias = d_activator/d_activator_input * d_activator_input/d_bias = d_activator/d_activator_input * 1
                // Instead of actually giving the sigmoid input to the activator derivative function, we give the
                // already-activated value, just because the calculation is simpler (sigmoid*(1-sigmoid)), and it's what we know: the node value
                derivative = layerTransforms[bStartLayer].Activator.ActivationDerivative(nodes[layer][bEndNode]);
            else
                // This bias has nothing to do with that node
                derivative = 0;
            biasDerivativeCache[(layer, node, bStartLayer, bEndNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(db__) = Sum{(dhi_)/(dhi-1_) * (dhi-1_)/(db__)}
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
            // endnode = activator(w*startnode + ...), so d_endnode/d_startnode = d_activator/d_activator_input * d_activator_input/d_startnode = d_activator/d_activator_input * w
            // Instead of actually giving the sigmoid input to the activator derivative function, we give the
            // already-activated value, just because the calculation is simpler (sigmoid*(1-sigmoid)), and it's what we know: the node value
            double derivative = layerTransforms[startLayer].Activator.ActivationDerivative(nodes[endLayer][endNode]) * layerTransforms[startLayer].Weights[endNode, startNode];
            nodeDerivativeCache[(endLayer, endNode, startLayer, startNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(dj__) = Sum{(dhi_)/(dhi-1_) * (dhi-1_)/(dj__)}
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
        NeuralNet clone = new(nodes[0].Count, nodes[1].Count, nodes[^1].Count, layerTransforms.Length - 1);
        for (int i = 0; i < layerTransforms.Length; i++)
            clone.layerTransforms[i] = layerTransforms[i].Clone();
        for (int i = 0; i < nodes.Length; i++)
            clone.nodes[i] = nodes[i].Clone();
        return clone;
    }
}