using System;
using System.Collections.Generic;
using System.Threading;
namespace LPE {
    public class ThreadInstance {
        static ObjectPool<ThreadInstance> pool = new ObjectPool<ThreadInstance>(() => new ThreadInstance());
        static List<ThreadInstance> used = new List<ThreadInstance>();
        public static ThreadInstance Get() {
            // return finished threads to pool
            for (int i = 0; i < used.Count; i++) {
                var t = used[i];
                if (t.done) {
                    pool.Return(t);
                    used.RemoveAt(i);
                    i--;
                }
            }

            // get from pool
            var ti = pool.Get();
            used.Add(ti);
            return ti;
        }


        public bool done => task == null;
        Thread thread;
        Action task;
        Action onComplete;
        EventWaitHandle ewh;



        private ThreadInstance() {
            ewh = new EventWaitHandle(false, EventResetMode.ManualReset);

            thread = new Thread(
                () => {
                    while (true) {
                        ewh.WaitOne();
                        try {
                            task?.Invoke();
                        }
                        catch {
                            // TODO Throw in main thread
                        }
                        // finish
                        onComplete?.Invoke();

                        //reset
                        ewh.Reset();
                        task = null;
                        onComplete = null;
                    }
                }
            );
            thread.Start();
        }

        public void SetTask(Action task, Action onComplete = null) {
            if (!done) {
                throw new InvalidOperationException("Thread Instance already has task");
            }
            this.task = task;
            this.onComplete = onComplete;
            ewh.Set();
        }
    }
}