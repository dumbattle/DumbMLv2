using System;
using UnityEngine;
using DumbML.RL;
using System.Diagnostics;

namespace FallingRocks {
    public class FallingRocksMain : MonoBehaviour {
        public GameDisplayBehaviour displayBehaviour;
        public int speed;
        Game game;
        A2CTrainer trainer;

        Stopwatch sw = new Stopwatch();
        void Start() {
            game = new Game(GameSettings.Default());

            trainer = new FRA2C(game);
        }

        void Update() {
            displayBehaviour.DisplayGame(game);
            sw.Reset();
            sw.Start();
            for (int i = 0; i < speed; i++) {
                trainer.Step();
                if (sw.ElapsedMilliseconds > 1000) {
                    break;
                }
            }
        }

        private void OnDestroy() {
            trainer.Dispose();
        }
    }

}
