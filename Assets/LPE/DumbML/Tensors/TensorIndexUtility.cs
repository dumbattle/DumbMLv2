using System;


namespace DumbML {
    public static class TensorIndexUtility {
        public static void CheckIndex(this Tensor t, params int[] indexes) {
            if (indexes.Length != t.shape.Length) {
                throw new ArgumentException($"Index has invalid number of parameters. Got {indexes.Length} Expected {t.shape.Length}");
            }

            for (int i = 0; i < indexes.Length; i++) {
                if (indexes[i] < 0 || indexes[i] >= t.shape[i]) {
                    throw new ArgumentOutOfRangeException("indexes", $"Shape: {t.shape.ContentString()}  Index:{indexes.ContentString()}");
                }
            }
        }

        public static void CheckIndex(this Tensor t, int a) {
            if (1 != t.shape.Length) {
                throw new ArgumentException($"Index has invalid number of parameters. Got {1} Expected {t.shape.Length}");
            }

            if (a < 0 || a >= t.shape[0]) {
                throw new ArgumentOutOfRangeException("indexes", $"Shape: {t.shape.ContentString()}  Index:[{a}]");
            }
        }
        public static void CheckIndex(this Tensor t, int a, int b) {
            if (2 != t.shape.Length) {
                throw new ArgumentException($"Index has invalid number of parameters. Got {2} Expected {t.shape.Length}");
            }

            bool invalid =
                a < 0 || a >= t.shape[0] ||
                b < 0 || b >= t.shape[1]
            ;
            if (invalid) {
                throw new ArgumentOutOfRangeException("indexes", $"Shape: {t.shape.ContentString()}  Index:[{a}, {b}]");
            }
        }
        public static void CheckIndex(this Tensor t, int a, int b, int c) {
            if (3 != t.shape.Length) {
                throw new ArgumentException($"Index has invalid number of parameters. Got {3} Expected {t.shape.Length}");
            }

            bool invalid =
                a < 0 || a >= t.shape[0] ||
                b < 0 || b >= t.shape[1] ||
                c < 0 || c >= t.shape[2]
            ;
            if (invalid) {
                throw new ArgumentOutOfRangeException("indexes", $"Shape: {t.shape.ContentString()}  Index:[{a}, {b}, {c}]");
            }
        }

        public static void CheckIndex(this Tensor t, int a, int b, int c, int d) {
            if (4 != t.shape.Length) {
                throw new ArgumentException($"Index has invalid number of parameters. Got {3} Expected {t.shape.Length}");
            }

            bool invalid =
                a < 0 || a >= t.shape[0] ||
                b < 0 || b >= t.shape[1] ||
                c < 0 || c >= t.shape[2] ||
                d < 0 || d >= t.shape[3]
            ;
            if (invalid) {
                throw new ArgumentOutOfRangeException("indexes", $"Shape: {t.shape.ContentString()}  Index:[{a}, {b}, {c}, {d}]");
            }
        }



        public static int GetIndex(this Tensor t, params int[] indexes) {
            int result = 0;
            int scale = 1;

            for (int i = indexes.Length - 1; i >= 0; i--) {
                result += indexes[i] * scale;
                scale *= t.shape[i];
            }

            return result;
        }
        public static int GetIndex(this Tensor t, int a) {
            return a;
        }
        public static int GetIndex(this Tensor t, int a, int b) {
            return a * t.shape[1] + b;
        }
        public static int GetIndex(this Tensor t, int a, int b, int c) {
            return (a * t.shape[1] + b) * t.shape[2] + c;
        }
        public static int GetIndex(this Tensor t, int a, int b, int c, int d) {
            return ((a * t.shape[1] + b) * t.shape[2] + c) * t.shape[3] + d;
        }

    }
}

