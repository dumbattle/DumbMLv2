using System;

namespace DumbML.RL {
    public abstract class RLGame {
        public abstract void GetState(Tensor[] newStateResults);
        public abstract float Step(Tensor[] actions);
        public abstract bool IsDone();
        public abstract void Reset();

        public static RLGame Create(Action<Tensor[]> getState, Func<Tensor[], float> step, Func<bool> isDone, Action reset) {
            var result = new ExplicitGame();
            result._getState = getState;
            result._step = step;
            result._isDone = isDone;
            result._reset = reset;
            return result;
        }
        class ExplicitGame : RLGame {
            public Action _reset;
            public Action<Tensor[]> _getState;
            public Func<bool> _isDone;
            public Func<Tensor[], float> _step;


            public override void GetState(Tensor[] newStateResults) {
                _getState(newStateResults);
            }

            public override bool IsDone() {
                return _isDone();
            }

            public override void Reset() {
                _reset();
            }

            public override float Step(Tensor[] actions) {
                return _step(actions);
            }
        }
    }
}
