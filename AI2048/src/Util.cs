using AI2048.Deep;
using MathNet.Numerics.LinearAlgebra;

namespace AI2048;

public static class Util
{
    /// <summary>
    /// Perform dot product on two dictionaries
    /// </summary>
    /// <param name="d1"></param>
    /// <param name="d2"></param>
    /// <returns></returns>
    public static double DictionaryDotProduct<T>(Dictionary<T, double> d1, Dictionary<T, double> d2) where T : class
    {
        double sum = 0;
        foreach(T key in d1.Keys)
        {
            double d2Val = d2.TryGetValue(key, out double value) ? value : 0;
            sum += d2Val * d1[key];
        }
        return sum;
    }

    public static void SaveTrainingState(this NeuralNet neuralNet, string filename)
    {
        StreamWriter streamWriter = new(filename, false);

        // Meta data
        streamWriter.WriteLine(neuralNet.NumInputNodes);
        streamWriter.WriteLine(neuralNet.NumMiddleNodes);
        streamWriter.WriteLine(neuralNet.NumOutputNodes);
        streamWriter.WriteLine(neuralNet.NumMiddleLayers);

        // Weights
        foreach (Matrix<double> weightMatrix in neuralNet.LayerTransforms.Select(t => t.Weights))
            for (int i = 0; i < weightMatrix.RowCount; i++)
                for (int j = 0; j < weightMatrix.ColumnCount; j++)
                    streamWriter.WriteLine(weightMatrix.At(i, j));

        // Biases
        foreach (Vector<double> biasVector in neuralNet.LayerTransforms.Select(t => t.Biases))
            for (int i = 0; i < biasVector.Count; i++)
                streamWriter.WriteLine(biasVector[i]);

        streamWriter.Close();
    }

    public static NeuralNet GetNeuralNetFromFile(string filename)
    {
        StreamReader streamReader= new(filename);

        int numInputNodes = streamReader.ParseIntFromNextLine();
        int numMiddleNodes = streamReader.ParseIntFromNextLine();
        int numOutputNodes = streamReader.ParseIntFromNextLine();
        int numMiddleLayers = streamReader.ParseIntFromNextLine();

        NeuralNet neuralNet = new(numInputNodes, numMiddleNodes, numOutputNodes, numMiddleLayers);

        // Weights
        int startLayer = 0;
        foreach (Matrix<double> weightMatrix in neuralNet.LayerTransforms.Select(t => t.Weights))
        {
            for (int i = 0; i < weightMatrix.RowCount; i++)
                for (int j = 0; j < weightMatrix.ColumnCount; j++)
                    neuralNet.SetWeight(startLayer, j, i, streamReader.ParseDoubleFromNextLine());
            startLayer++;
        }

        // Biases
        startLayer = 0;
        foreach (Vector<double> biasVector in neuralNet.LayerTransforms.Select(t => t.Biases))
        {
            for (int i = 0; i < biasVector.Count; i++)
                neuralNet.SetBias(startLayer, i, streamReader.ParseDoubleFromNextLine());
            startLayer++;
        }

        streamReader.Close();

        return neuralNet;
    }

    private static int ParseIntFromNextLine(this StreamReader streamReader) =>
        int.Parse(streamReader.ReadLine() ?? string.Empty);

    private static double ParseDoubleFromNextLine(this StreamReader streamReader) =>
        double.Parse(streamReader.ReadLine() ?? string.Empty);
}