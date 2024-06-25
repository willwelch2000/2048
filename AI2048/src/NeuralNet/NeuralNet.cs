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
    
    /// <summary>
    /// objects defining how to move from one layer to the next
    /// </summary>
    private readonly LayerTransform[] layerTransforms;

    /// <summary>
    /// All nodes in the network
    /// </summary>
    private readonly Vector<double>[] nodes;

    // caches for derivative calculations of a node with respect to something--mostly deprecated

    private Dictionary<(int layer, int node, int wStartLayer, int wEndNode, int wStartNode), double> nodeToWeightDerivativeCache;
    private Dictionary<(int layer, int node, int bStartLayer, int bEndNode), double> nodeToBiasDerivativeCache;
    private Dictionary<(int endLayer, int endNode, int startLayer, int startNode), double> nodeToNodeDerivativeCache;

    /// </summary>
    /// Cache for derivatives of loss with respect to weights--used for accessor function
    /// The same shape as the weights themselves, so that they can be added
    /// </summary>
    private Matrix<double>[]? lossToWeightDerivativeCache = null;

    /// </summary>
    /// Cache for derivatives of loss with respect to biases--used for accessor function
    /// The same shape as the biases themselves, so that they can be added
    /// </summary>
    private Vector<double>[]? lossToBiasDerivativeCache = null;

    // // // constructors

    /// <summary>
    /// Constructor that uses Sigmoid activator by default
    /// </summary>
    /// <param name="numInputNodes">Number of nodes for input layer</param>
    /// <param name="numMiddleNodes">Number of nodes for middle layers</param>
    /// <param name="numOutputNodes">Number of nodes for output layer</param>
    /// <param name="numMiddleLayers">Number of middle layers</param>
    public NeuralNet(int numInputNodes, int numMiddleNodes, int numOutputNodes, int numMiddleLayers) : this(numInputNodes, numMiddleNodes, numOutputNodes, numMiddleLayers, new Sigmoid()) {}

    /// <summary>
    /// Main constructor
    /// </summary>
    /// <param name="numInputNodes">Number of nodes for input layer</param>
    /// <param name="numMiddleNodes">Number of nodes for middle layers</param>
    /// <param name="numOutputNodes">Number of nodes for output layer</param>
    /// <param name="numMiddleLayers">Number of middle layers</param>
    /// <param name="activator"></param>
    public NeuralNet(int numInputNodes, int numMiddleNodes, int numOutputNodes, int numMiddleLayers, IActivationFunction primaryActivator, int? randomSeed = null)
    {
        // Initialize layer transforms
        layerTransforms = new LayerTransform[numMiddleLayers + 1];

        // First transform
        layerTransforms[0] = new(
            randomSeed is null ? Matrix<double>.Build.Random(numMiddleLayers > 0 ? numMiddleNodes : numOutputNodes, numInputNodes) : Matrix<double>.Build.Random(numMiddleLayers > 0 ? numMiddleNodes : numOutputNodes, numInputNodes, randomSeed ?? 0),
            randomSeed is null ? Vector<double>.Build.Random(numMiddleNodes) : Vector<double>.Build.Random(numMiddleNodes, randomSeed ?? 0),
            primaryActivator);

        // Last transform
        layerTransforms[^1] = new(
            randomSeed is null ? Matrix<double>.Build.Random(numOutputNodes, numMiddleLayers > 0 ? numMiddleNodes : numInputNodes) : Matrix<double>.Build.Random(numOutputNodes, numMiddleLayers > 0 ? numMiddleNodes : numInputNodes, randomSeed ?? 0),
            randomSeed is null ? Vector<double>.Build.Random(numOutputNodes) : Vector<double>.Build.Random(numOutputNodes, randomSeed ?? 0),
            primaryActivator);

        // All other transforms
        for (int i = 1; i < numMiddleLayers; i++)
            layerTransforms[i] = new(
                randomSeed is null ? Matrix<double>.Build.Random(numMiddleNodes, numMiddleNodes) : Matrix<double>.Build.Random(numMiddleNodes, numMiddleNodes, randomSeed ?? 0),
                randomSeed is null ? Vector<double>.Build.Random(numMiddleNodes) : Vector<double>.Build.Random(numMiddleNodes, randomSeed ?? 0),
                primaryActivator);

        // Initialize nodes
        nodes = new Vector<double>[numMiddleLayers + 2];
        nodes[0] = Vector<double>.Build.Dense(numInputNodes);
        nodes[^1] = Vector<double>.Build.Dense(numOutputNodes);
        for (int i = 1; i < numMiddleLayers + 1; i++)
            nodes[i] = Vector<double>.Build.Dense(numMiddleNodes);

        // Initialize cache dictionaries and node derivative array
        nodeToWeightDerivativeCache = [];
        nodeToBiasDerivativeCache = [];
        nodeToNodeDerivativeCache = [];
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

    // Shortcuts to see shape of net

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
        ResetDerivativeCaches();
        nodes[0] = input.Clone();
        for (int i = 0; i < layerTransforms.Length; i++)
            layerTransforms[i].TransformLayer(nodes[i], nodes[i + 1]);

        return nodes.Last().Clone();
    }

    /// <summary>
    /// Set a specific weight
    /// </summary>
    /// <param name="startLayer"></param>
    /// <param name="startNode"></param>
    /// <param name="endNode"></param>
    /// <param name="value"></param>
    public void SetWeight(int startLayer, int startNode, int endNode, double value)
    {
        layerTransforms[startLayer].Weights[endNode, startNode] = value;
        ResetDerivativeCaches();
    }

    /// <summary>
    /// Set a specific bias
    /// </summary>
    /// <param name="startLayer"></param>
    /// <param name="endNode"></param>
    /// <param name="value"></param>
    public void SetBias(int startLayer, int endNode, double value)
    {
        layerTransforms[startLayer].Biases[endNode] = value;
        ResetDerivativeCaches();
    }

    /// <summary>
    /// Set the activator for a layer transform
    /// </summary>
    /// <param name="startLayer"></param>
    /// <param name="activator"></param>
    public void SetActivator(int startLayer, IActivationFunction activator)
    {
        layerTransforms[startLayer].Activator = activator;
        ResetDerivativeCaches();
    }

    /// <summary>
    /// Given a single set of input values and its desired output, adjust the weights accordingly
    /// </summary>
    /// <param name="input">input layer</param>
    /// <param name="compare">desired output</param>
    public void PerformGradientDescent(Vector<double> input, Vector<double> compare)
    {
        (Matrix<double>[] weightDerivatives, Vector<double>[] biasDerivatives) = CalculateLossDerivatives(input, compare);

        for (int i = 0; i < layerTransforms.Length; i++)
        {
            layerTransforms[i].Weights -= Alpha * weightDerivatives[i];
            layerTransforms[i].Biases -= Alpha * biasDerivatives[i];
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
    /// Node derivatives are mostly deprecated because it's slower
    /// </summary>
    /// <param name="layer">layer of the numerator node for the derivative. For example, with dy1/dw111, y is the layer</param>
    /// <param name="node">which node in the layer for the numerator for the derivative</param>
    /// <param name="wStartLayer">the start layer of the weight under examination</param>
    /// <param name="wEndNode">the end node of the weight under examination</param>
    /// <param name="wStartNode">the start node of the weight under examination</param>
    /// <returns></returns>
    public double GetNodeToWeightDerivative(int layer, int node, int wStartLayer, int wEndNode, int wStartNode)
    {
        // Already cached
        if (nodeToWeightDerivativeCache.TryGetValue((layer, node, wStartLayer, wEndNode, wStartNode), out double value))
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
            nodeToWeightDerivativeCache[(layer, node, wStartLayer, wEndNode, wStartNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(dw___) = Sum{(dhi_)/(dhi-1_) * (dhi-1_)/(dw___)}
        if (wStartLayer + 1 < layer)
        {
            int prevLayerLength = nodes[layer - 1].Count;
            double derivative = Enumerable.Range(0, prevLayerLength).Select(i => GetNodeToNodeDerivative(layer, node, layer - 1, i) * GetNodeToWeightDerivative(layer - 1, i, wStartLayer, wEndNode, wStartNode)).Sum();
            // double derivative = Enumerable.Range(0, prevLayerLength).Select(i => nodeDerivatives[layer][node, i] * WeightDerivative(layer - 1, i, wStartLayer, wEndNode, wStartNode)).Sum();
            nodeToWeightDerivativeCache[(layer, node, wStartLayer, wEndNode, wStartNode)] = derivative;
            return derivative;
        }

        return 0;
    }

    /// <summary>
    /// Get derivative of node with respect to a bias
    /// To work properly, the GetOutputValues() function must be called first
    /// Node derivatives are mostly deprecated because it's slower
    /// </summary>
    /// <param name="layer">layer of the numerator node for the derivative</param>
    /// <param name="node">which node in the target layer for the derivative</param>
    /// <param name="bStartLayer">the starting layer (layer before the one that it's used to calculate) of the bias that is the denominator</param>
    /// <param name="bEndNode">the node that this bias is involved in calculating</param>
    /// <returns></returns>
    public double GetNodeToBiasDerivative(int layer, int node, int bStartLayer, int bEndNode)
    {
        // Already cached
        if (nodeToBiasDerivativeCache.TryGetValue((layer, node, bStartLayer, bEndNode), out double value))
            return value;

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
            nodeToBiasDerivativeCache[(layer, node, bStartLayer, bEndNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(db__) = Sum{(dhi_)/(dhi-1_) * (dhi-1_)/(db__)}
        if (bStartLayer + 1 < layer)
        {
            int prevLayerLength = nodes[layer - 1].Count;
            double derivative = Enumerable.Range(0, prevLayerLength).Select(i => GetNodeToNodeDerivative(layer, node, layer - 1, i) * GetNodeToBiasDerivative(layer - 1, i, bStartLayer, bEndNode)).Sum();
            // double derivative = Enumerable.Range(0, prevLayerLength).Select(i => nodeDerivatives[layer][node, i] * BiasDerivative(layer - 1, i, bStartLayer, bEndNode)).Sum();
            nodeToBiasDerivativeCache[(layer, node, bStartLayer, bEndNode)] = derivative;
            return derivative;
        }

        return 0;
    }

    /// <summary>
    /// Get derivative of node with respect to another node
    /// To work properly, the GetOutputValues() function must be called first
    /// Node derivatives are mostly deprecated because it's slower
    /// </summary>
    /// <param name="endLayer">layer of the numerator node for the derivative</param>
    /// <param name="endNode">which node in the target layer for the derivative</param>
    /// <param name="startLayer">layer of the denominator node</param>
    /// <param name="startNode">which node in the layer for the denominator node</param>
    /// <returns></returns>
    public double GetNodeToNodeDerivative(int endLayer, int endNode, int startLayer, int startNode)
    {
        // Already cached
        if (nodeToNodeDerivativeCache.TryGetValue((endLayer, endNode, startLayer, startNode), out double value))
            return value;

        // Simple case for derivative
        if (endLayer == startLayer + 1)
        {
            // endnode = activator(w*startnode + ...), so d_endnode/d_startnode = d_activator/d_activator_input * d_activator_input/d_startnode = d_activator/d_activator_input * w
            // Instead of actually giving the sigmoid input to the activator derivative function, we give the
            // already-activated value, just because the calculation is simpler (sigmoid*(1-sigmoid)), and it's what we know: the node value
            double derivative = layerTransforms[startLayer].Activator.ActivationDerivative(nodes[endLayer][endNode]) * layerTransforms[startLayer].Weights[endNode, startNode];
            nodeToNodeDerivativeCache[(endLayer, endNode, startLayer, startNode)] = derivative;
            return derivative;
        }

        // Most cases--use chain rule. This derivative (dhi_)/(dj__) = Sum{(dhi_)/(dhi-1_) * (dhi-1_)/(dj__)}
        if (startLayer + 1 < endLayer)
        {
            int prevLayerLength = nodes[endLayer - 1].Count;
            double derivative = Enumerable.Range(0, prevLayerLength).Select(i => GetNodeToNodeDerivative(endLayer, endNode, endLayer - 1, i) * GetNodeToNodeDerivative(endLayer - 1, i, startLayer, startNode)).Sum();
            nodeToNodeDerivativeCache[(endLayer, endNode, startLayer, startNode)] = derivative;
            return derivative;
        }

        return 0;
    }

    /// <summary>
    /// Get derivative of loss function with respect to a weight
    /// To work properly, the CalculateLossDerivatives() function must be called first
    /// </summary>
    /// <param name="startLayer">layer of layer transformed by this weight (e.g. 1 means the weights going from layer 1 to 2)</param>
    /// <param name="endNode">which node in the target (end) layer for the weight</param>
    /// <param name="startNode">which node in the start layer for the weight</param>
    /// <returns></returns>
    public double? GetLossToWeightDerivative(int startLayer, int endNode, int startNode) =>
        lossToWeightDerivativeCache?[startLayer][endNode, startNode] ?? null;
    
    /// <summary>
    /// Get derivative of loss function with respect to a bias
    /// To work properly, the CalculateLossDerivatives() function must be called first
    /// </summary>
    /// <param name="startLayer">layer of layer transformed by this weight (e.g. 1 means the biases going from layer 1 to 2)</param>
    /// <param name="endNode">which node in the target (end) layer for the bias</param>
    /// <returns></returns>
    public double? GetLossToBiasDerivative(int startLayer, int endNode) =>
        lossToBiasDerivativeCache?[startLayer][endNode] ?? null;

    /// <summary>
    /// Uses matrix multiplication to efficiently calculate derivatives
    /// </summary>
    /// <param name="output">current output</param>
    /// <param name="compare">desired output</param>
    public (Matrix<double>[] weightDerivatives, Vector<double>[] biasDerivatives) CalculateLossDerivatives(Vector<double> input, Vector<double> compare)
    {
        // Store derivatives of weights and biases--both of these are the same shape of the weights/biases so that they can be added together
        Matrix<double>[] weightDerivatives = new Matrix<double>[nodes.Length - 1];
        Vector<double>[] biasDerivatives = new Vector<double>[nodes.Length - 1];

        Vector<double> output = GetOutputValues(input);

        // Calculate d_Loss/d_output as a row matrix, where v[0, 0] gives the derivative of the loss with respect to the first output node
        // We start with the last layer, this will get transformed as we go on
        Matrix<double> nodeDerivativeMatrix = (2 * (output - compare)).ToRowMatrix();

        // Work backwards, starting at 2nd-to-last layer since last layer is already done
        // In each loop, calculate the derivative of the (layer)th weights and then (layer)th nodes
        for (int layer = nodes.Length - 2; layer >= 0; layer--)
        {
            LayerTransform layerTransform = layerTransforms[layer];
            Vector<double> startNodes = nodes[layer];
            Vector<double> endNodes = nodes[layer + 1];
            int endLayerLength = endNodes.Count;

            // Calculate derivative of the current layer pre-activation (the derivative of the sums that get activated) as a row matrix
            Matrix<double> activationDerivativeMatrix = Matrix<double>.Build.Diagonal(endLayerLength, endLayerLength, i => layerTransform.Activator.ActivationDerivative(endNodes[i]));
            Matrix<double> preActivationNodeDerivativeMatrix = nodeDerivativeMatrix * activationDerivativeMatrix;

            // Get weight derivatives--this is the d_Loss/dw for each entry in the weight matrix--can be added to weight matrix
            // This is calculated by taking the derivative of the unactivated next layer (transposed to be a column) times this layer of nodes (as row)
            weightDerivatives[layer] = preActivationNodeDerivativeMatrix.TransposeThisAndMultiply(nodes[layer].ToRowMatrix());
            
            // Bias derivative matrix is just the same as the preactivation derivatives, but as a vector
            biasDerivatives[layer] = preActivationNodeDerivativeMatrix.Row(0);

            // Matrix representing Jacobian of activation function (m[i,j] = derivative of ith output with respect to jth input, dyi/dxj)
            // For a function like the activation functions, which just perform a function on each input by itself, it's a diagonal matrix
            nodeDerivativeMatrix = preActivationNodeDerivativeMatrix * layerTransform.Weights;
        }

        (lossToWeightDerivativeCache, lossToBiasDerivativeCache) = (weightDerivatives, biasDerivatives);

        return (weightDerivatives, biasDerivatives);
    }

    public void ResetDerivativeCaches()
    {
        nodeToWeightDerivativeCache = [];
        nodeToBiasDerivativeCache = [];
        nodeToNodeDerivativeCache = [];
        lossToWeightDerivativeCache = null;
        lossToBiasDerivativeCache = null;
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