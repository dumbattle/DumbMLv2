using System.Collections.Generic;
using LPE;

namespace DumbML.BLAS {
    public static class Utils {
        static ObjectPool<int[]> intArrPool = new ObjectPool<int[]>(() => new int[64]);
        static ObjectPool<List<int>> intListPool = new ObjectPool<List<int>>(() => new List<int>());
        public static int[] GetIntArr() {
            return intArrPool.Get();
        }
        public static List<int> GetIntList() {
            return intListPool.Get();
        }

        public static void Return(int[] arr) {
            intArrPool.Return(arr);
        }
        public static void Return(List<int> l) {
            intListPool.Return(l);
        }
    }

}
