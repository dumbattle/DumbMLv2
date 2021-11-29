using System;
using System.Threading;
namespace LPE {
    public class ThreadCompletionCallback : IThreadCompletionCallback {
        static ObjectPool<ThreadCompletionCallback> pool
            = new ObjectPool<ThreadCompletionCallback>(() => new ThreadCompletionCallback());

        public static ThreadCompletionCallback Get() {
            return pool.Get();
        }
        public Action SetDone;
        public bool done { get; set; }

        public EventWaitHandle waitHandle { get; private set; } = new EventWaitHandle(false, EventResetMode.ManualReset);

        private ThreadCompletionCallback() {
            SetDone = _OnComplete;
        }

        public void Start() {
            done = false;
            waitHandle.Reset();
        }
        void _OnComplete() {
            done = true;
            waitHandle.Set();
        }
        public void Return() {
            pool.Return(this);
        }
    }
}