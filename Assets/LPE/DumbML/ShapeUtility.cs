using System.Collections.Generic;

namespace DumbML {
    public static class ShapeUtility {
        public static bool SameShape(List<int> a, List<int> b) {
            if (a.Count != b.Count) {
                return false;
            }

            for (int i = 0; i < a.Count; i++) {
                if (a[i] != b[i]) {
                    return false;
                }
            }
            return true;
        }
        public static bool SameShape(int[] a, int[] b) {
            if (a.Length != b.Length) {
                return false;
            }

            for (int i = 0; i < a.Length; i++) {
                if (a[i] != b[i]) {
                    return false;
                }
            }
            return true;
        }
    }
}

