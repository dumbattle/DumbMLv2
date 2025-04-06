using System;
using UnityEngine;


namespace FreeSnake {
    [Serializable]
    public struct GameParameters {
        public Vector2 mapsize;
        public float moveSpeed;
        public float playerRadius;
        public float targetRadius;
        public float idleScore;
        public float targetScore;
        public int maxSteps;
    }
}