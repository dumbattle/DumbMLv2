using System.Threading;

namespace LPE {

    public interface IThreadCompletionCallback {
        EventWaitHandle waitHandle { get; }
        bool done { get; }

        void Return();
    }
}