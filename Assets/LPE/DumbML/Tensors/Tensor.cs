using System.Collections;
using System;
using UnityEngine;


namespace DumbML {
    public class Tensor {
        #region Indexers
        public float this[params int[] index] {
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
        public float this[int a] {
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
        public float this[int a, int b] {
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
        public float this[int a, int b, int c] {
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
        public float this[int a, int b, int c, int d] {
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


        public float[] data { get; private set; }
        public int[] shape { get; private set; }
        public int size => data.Length;


        public Tensor(params int[] shape) {
            this.shape = (int[])shape.Clone();

            int size = 1;

            foreach (var i in shape) {
                size *= i;
            }

            data = new float[size];
        }


        public static Tensor FromArray(Array A) {
            
            int[] shape = new int[A.Rank];
            for (int i = 0; i < A.Rank; i++) {
                shape[i] = A.GetLength(i);
            }
          
            Tensor result = new Tensor(shape);
            int index = 0;

            foreach (var val in A) {
                result.data[index] = Convert.ToSingle(val);
                index++;
            }

            return result;
        }
    }



}