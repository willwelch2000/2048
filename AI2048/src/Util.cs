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
            double d2Val = d2.ContainsKey(key) ? d2[key] : 0;
            sum += d2Val * d1[key];
        }
        return sum;
    }
}