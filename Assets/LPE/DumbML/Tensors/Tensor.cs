﻿using System.Collections.Generic;


namespace DumbML {
    public abstract class Tensor {
        public abstract DType dtype { get; }
        public int[] shape { get; protected set; }
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
    }
}