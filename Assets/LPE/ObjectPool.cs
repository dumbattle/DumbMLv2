using System.Collections.Generic;
using System;

namespace LPE {
    public class ObjectPool {
        //*********************************************************************************************
        // Singleton Helper
        //*********************************************************************************************
        static class Singleton<T> where T : class, new() {
            public static ObjectPool<T> pool = new ObjectPool<T>(() => new T());
        }

        public static T Get<T>() where T : class, new() {
            return Singleton<T>.pool.Get();
        }

        public static void Return<T>(T item) where T : class, new() {
            Singleton<T>.pool.Return(item);
        }


        public static List<T> GetList<T>() {
            var result = Singleton<List<T>>.pool.Get();
            result.Clear();
            return result;
        }

        public static void ReturnList<T>(List<T> item) {
            item.Clear();
            Singleton<List<T>>.pool.Return(item);
        }



        public static Dictionary<TKey, TVal> GetDictionary<TKey, TVal>() {
            var result = Singleton<Dictionary<TKey, TVal>>.pool.Get();
            result.Clear();
            return result;
        }

        public static void ReturnDictionary<TKey, TVal>(Dictionary<TKey, TVal> item) {
            item.Clear();
            Singleton<Dictionary<TKey, TVal>>.pool.Return(item);
        }


        public static HashSet<T> GetHashSet<T>() {
            var result = Singleton<HashSet<T>>.pool.Get();
            result.Clear();
            return result;
        }

        public static void ReturnHashSet<T>(HashSet<T> item) {
            item.Clear();
            Singleton<HashSet<T>>.pool.Return(item);
        }
        //*********************************************************************************************
        // Reminder Helper
        //*********************************************************************************************
        static HashSet<string> reminders = new HashSet<string>();
        public static void SetReminder(string id) {
            if (reminders.Contains(id)) {
                return;
            }
            reminders.Add(id);

            UnityEngine.Debug.LogWarning($"Rember to implement object pool for '{id}'");

        }
    }

    public class ObjectPool<T> where T : class {
        Dictionary<T, Item> returnDict = new Dictionary<T, Item>();
        Func<T> _constructor;
        LinkedList<Item> freeItems = new LinkedList<Item>();
        int warningCount;


        public ObjectPool(Func<T> objCreater, int warningCount = 1000) {
            _constructor = objCreater;
            this.warningCount = warningCount;
        }

        public T Get() {
            if (freeItems.First != null) {
                var n = freeItems.First;

                freeItems.RemoveFirst();

                return n.Value.obj;
            }

            var newItem = CreateItem();

            return Get();
        }

        public void Return(T t) {
            var n = returnDict[t].node;
            freeItems.AddLast(n);
        }

        Item CreateItem() {
            T t = _constructor();
            Item i = new Item(t);

            returnDict.Add(t, i);
            freeItems.AddLast(i.node);

            if (returnDict.Count % warningCount == 0) {
                UnityEngine.Debug.LogWarning($"ObjectPool<{typeof(T).Name}> capacity reached {returnDict.Count}");
            }
            return i;
        }


        class Item {
            public T obj;
            public LinkedListNode<Item> node;

            public Item(T t) {
                obj = t;
                node = new LinkedListNode<Item>(this);
            }
        }
    }
}