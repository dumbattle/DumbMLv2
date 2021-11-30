namespace DumbML {

    public interface ITensorBuffer:System.IDisposable {
        int[] shape { get; }
        int size { get; }
        Device device { get; }

        void SetShape(int[] shape);

        void CopyFrom(Tensor t);
        void CopyTo(Tensor t);
    }
}
