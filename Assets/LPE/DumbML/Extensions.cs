﻿using System.Collections.Generic;
using System.Text;
using Unity.Collections;


namespace DumbML {
    public static class Extensions {

        public static void Shuffle<T>(this T[] array) {
            var rng = new System.Random();

            int n = array.Length;
            while (n > 1) {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
        public static float Abs(this float a) {
            return a > 0 ? a : -a;
        }
        public static float Sqr(this float a) {
            return a * a;
        }
        public static float Sqrt(this float a) {
            return (float)System.Math.Sqrt(a);
        }
        public static float Sign(this float a) {
            return a >= 0 ? 1 : -1;
        }
        public static float Clamp(this float a, float min, float max) {
            if (a < min) {
                return min;
            }
            if (a > max) {
                return max;
            }

            return a;
        }
        public static float Lerp(this float a, float b, float t, bool bound = true) {
            if (bound) {
                t = t.Clamp(0, 1);
            }
            float result = a * (1 - t) + b * t;
            return result;
        }


        public static int Clamp(this int a, int min, int max) {
            if (a < min) {
                return min;
            }
            if (a > max) {
                return max;
            }

            return a;
        }

        public static T Last<T>(this T[] a) {
            return a[a.Length - 1];
        }
        public static T Last<T>(this List<T> a) {
            return a[a.Count - 1];
        }
        public static bool Contains(this int[] a, int item) {
            foreach (var t in a) {
                if (t == item) {
                    return true;
                }
            }
            
            return false;
        }
        public static bool CompareContents<T>(this T[] a, T[] b) {
            if (a.Length != b.Length) {
                return false;
            }

            for (int i = 0; i < a.Length; i++) {
                if (Comparer<T>.Default.Compare(a[i], b[i]) != 0) {
                    return false;
                }
            }

            return true;
        }

        public static string ContentString<T>(this T[] t) {
            if (t.Length == 0) {
                return "[ ]";
            }
            StringBuilder sb = new StringBuilder();

            sb.Append("[");


            sb.Append(t[0]);
            for (int i = 1; i < t.Length; i++) {

                sb.Append(", " + t[i]);
            }

            sb.Append("]");
            return sb.ToString();
        }

        public static string ContentString<T>(this List<T> t) {
            StringBuilder sb = new StringBuilder();

            sb.Append("[");


            sb.Append(t[0]);
            for (int i = 1; i < t.Count; i++) {

                sb.Append(", " + t[i]);
            }

            sb.Append("]");
            return sb.ToString();
        }

        public static string ContentString<T>(this NativeArray<T> t) where T : struct {
            StringBuilder sb = new StringBuilder();

            sb.Append("[");


            sb.Append(t[0]);
            for (int i = 1; i < t.Length; i++) {

                sb.Append(", " + t[i]);
            }

            sb.Append("]");
            return sb.ToString();
        }


        public static void Deconstruct<T1, T2>(this KeyValuePair<T1, T2> src, out T1 key, out T2 value) {
            key = src.Key;
            value = src.Value;
        }
    }
}

