using System;
using System.Collections.Generic;
using System.Threading;



namespace LPE {
    public class MultiThreadCompletionCallback : IThreadCompletionCallback {
        static ObjectPool<MultiThreadCompletionCallback> pool
            = new ObjectPool<MultiThreadCompletionCallback>(() => new MultiThreadCompletionCallback());

        public static MultiThreadCompletionCallback Get() {
            return pool.Get();
        }

        public bool done => count == 0;

        public EventWaitHandle waitHandle { get; } = new EventWaitHandle(false, EventResetMode.ManualReset);

        int count = 0;

        public Action OnOneTaskDone;
        public void Start(int count) {
            this.count = count;
            waitHandle.Reset();
        }
        private MultiThreadCompletionCallback() {
            OnOneTaskDone = () => {
                Interlocked.Decrement(ref count);
                if (Interlocked.CompareExchange(ref count, 0, 0) == 0) {
                    waitHandle.Set();
                }
            };
        }

        public void Return() {
            pool.Return(this);
        }
    }
    public class ForIterationDelegate {
        static ObjectPool<ForIterationDelegate> pool = new ObjectPool<ForIterationDelegate>(() => new ForIterationDelegate());
        static List<ForIterationDelegate> used = new List<ForIterationDelegate>();


        public static ForIterationDelegate Get() {
            // return finished threads to pool
            for (int i = 0; i < used.Count; i++) {
                var t = used[i];
                if (t.done) {
                    pool.Return(t);
                    used.RemoveAt(i);
                    i--;
                }
            }

            var result = pool.Get();
            used.Add(result);
            return result;
        }


        public Action action { get; private set; }
        bool done => _action == null;
        int _i;
        Action<int> _action;


        private ForIterationDelegate() {
            action =
                () => {
                    _action(_i);
                    _action = null;
                };
        }

        public void SetInfo(Action<int> action, int i) {
            _action = action;
            _i = i;
        }

    }
}