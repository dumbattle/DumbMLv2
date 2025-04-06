using System.Collections.Generic;
using System.Text;
using System;

namespace DumbML {


    public abstract class Tensor {
        public abstract DType dtype { get; }
        public int[] shape { get; protected set; }
        public int rank => shape.Length;

        public abstract void CopyFrom(Tensor src, bool ignoreShape = false);

        public static Tensor Get(DType type, params int[] shape) {
            if (type == DType.Int) {
                return new IntTensor(shape);
            }
            if (type == DType.Float) {
                return new FloatTensor(shape);
            }
            if (type == DType.Bool) {
                return new BoolTensor(shape);
            }

            return null;
        }
    }
    public abstract class Tensor<T> : Tensor {
        #region Indexers
        public T this[params int[] index] {
            get {
                this.CheckIndex(index);
                int i = this.GetIndex(index);

                return data[i];
            }
            set {
                this.CheckIndex(index);
                int i = this.GetIndex(index);
                this.data[i] = value;
            }
        }
        public T this[int a] {
            get {
                this.CheckIndex(a);
                int i = this.GetIndex(a);

                return data[i];
            }
            set {
                this.CheckIndex(a);
                int i = this.GetIndex(a);
                this.data[i] = value;
            }
        }
        public T this[int a, int b] {
            get {
                this.CheckIndex(a, b);
                int i = this.GetIndex(a, b);

                return data[i];
            }
            set {
                this.CheckIndex(a, b);
                int i = this.GetIndex(a, b);
                this.data[i] = value;
            }
        }
        public T this[int a, int b, int c] {
            get {
                this.CheckIndex(a, b, c);
                int i = this.GetIndex(a, b, c);

                return data[i];
            }
            set {
                this.CheckIndex(a, b, c);
                int i = this.GetIndex(a, b, c);
                this.data[i] = value;
            }
        }
        public T this[int a, int b, int c, int d] {
            get {
                this.CheckIndex(a, b, c, d);
                int i = this.GetIndex(a, b, c, d);

                return data[i];
            }
            set {
                this.CheckIndex(a, b, c, d);
                int i = this.GetIndex(a, b, c, d);
                this.data[i] = value;
            }
        }
        #endregion Indexers

        public T[] data { get; private set; }
        public int size => data.Length;

        public Tensor(params int[] shape) {
            this.shape =(int[])shape.Clone();

            int size = 1;

            foreach (var i in shape) {
                size *= i;
            }

            data = new T[size];
        }

        public static Tensor<T> Get(params int[] shape) {
            if (typeof(T) == typeof(int)) {
                return new IntTensor(shape) as Tensor<T>;
            }
            if (typeof(T) == typeof(float)) {
                return new FloatTensor(shape) as Tensor<T>;
            }
            if (typeof(T) == typeof(bool)) {
                return new BoolTensor(shape) as Tensor<T>;
            }

            return null;
        }

        public override void CopyFrom(Tensor src, bool ignoreShape = false) {
            if (!ignoreShape) {
                if (!ShapeUtility.SameShape(shape, src.shape)) {
                    throw new ArgumentException(
                        $"Can't copy tensors with different shapes." +
                        $"\nSource: {src.shape.ContentString()}" +
                        $"\nDestination: {shape.ContentString()}");
                }
            }

            if (src is Tensor<T> tt) {
                Array.Copy(tt.data, data, data.Length);
            }
        }
      
        public override string ToString() {
            var result = new StringBuilder();

            result.Append($"{GetType().Name}");

            // shape
            result.Append($" (");

            for (int i = 0; i < rank; i++) {
                result.Append($"{shape[i]}");

                if (i != rank - 1) {
                    result.Append($", ");
                }
            }

            result.Append($")\n");
            string indent = "";

            // data
            for (int i = 0; i < size; i++) {
                int stride = 1;
                foreach (var d in shape) {
                    stride *= d;
                }

                for (int d = 0; d < rank; d++) {
                    if (i % stride == 0) {
                        result.Append("[");
                        if (d != rank - 1) {
                            indent += " ";
                        }
                    }
                  
                    stride /= shape[d];
                }

                result.Append(data[i].ToString(/*"N2"*/));

                bool brace = false;
                stride = 1;

                for (int d = rank - 1; d >= 0; d--) {
                    stride *= shape[d];
                    if ((i + 1) % stride == 0) {
                        result.Append("]");
                        if (d != rank - 1) {
                            indent = indent.Substring(0, indent.Length - 1);
                        }
                        else {
                        }

                        brace = true;
                    }
                    else {
                        break;
                    }
                }
              
                if (!brace) {
                    result.Append(", ");
                }
                else {
                    result.Append("\n" + indent);
                }
            }
            return result.ToString();
        }
    }
}