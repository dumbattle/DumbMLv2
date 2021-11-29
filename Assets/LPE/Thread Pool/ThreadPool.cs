using System;
using Unity.Profiling;
namespace LPE {
    public static class ThreadPool {

        public static IThreadCompletionCallback StartWithCallback(Action action) {
            ThreadCompletionCallback cb = ThreadCompletionCallback.Get();
            ThreadInstance ti = ThreadInstance.Get();
            cb.Start();
            ti.SetTask(action, cb.SetDone);

            return cb;
        }

        public static void Start(Action action) {
            ThreadInstance ti = ThreadInstance.Get();
            ti.SetTask(action);
        }


        public static IThreadCompletionCallback ForWithCallback(int startInclusive, int endExclusive, Action<int> body) {
            MultiThreadCompletionCallback cb = MultiThreadCompletionCallback.Get();
            cb.Start(endExclusive - startInclusive);

            for (int i = startInclusive; i < endExclusive; i++) {
                ThreadInstance ti = ThreadInstance.Get();
                ForIterationDelegate fid = ForIterationDelegate.Get();
                fid.SetInfo(body, i);
                ti.SetTask(fid.action, cb.OnOneTaskDone);
            }

            return cb;
        }
        public static void For(int startInclusive, int endExclusive, Action<int> body) {
            for (int i = startInclusive; i < endExclusive; i++) {
                ThreadInstance ti = ThreadInstance.Get();
                ForIterationDelegate fid = ForIterationDelegate.Get();
                fid.SetInfo(body, i);
                ti.SetTask(fid.action);
            }
        }
    }

    public static class ThreadPoolUtility {
        public static bool CheckAndReturn(this IThreadCompletionCallback cb) {
            bool done = cb.done;

            if (done) {
                cb.Return();
            }

            return done;
        }
        public static bool Wait(this IThreadCompletionCallback cb, int timeout = -1) {
            return cb.waitHandle.WaitOne(timeout);
        }
    }
}