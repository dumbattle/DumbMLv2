using DumbML;
using UnityEngine;


namespace FreeSnake {
    public class Game {
        public GameParameters parameters;
        public Vector2 playerPosition;
        public Vector2 targetPosition;
        public bool done;

        int steps;
        public Game(GameParameters parameters) {
            this.parameters = parameters;
        }

        public void Reset() {
            playerPosition = parameters.mapsize / 2;
            SetTargetPosition();
            done = false;
            steps = 0;
        }

        public float Update(Vector2 playerMovement) {
            if (done) { 
                return 0;
            }
            if (steps >= parameters.maxSteps) {
                done = true;
                return 0;
            }
            steps++;
            playerPosition += playerMovement.normalized * parameters.moveSpeed;

            done = 
                playerPosition.x < 0 ||
                playerPosition.y < 0 || 
                playerPosition.x > parameters.mapsize.x ||
                playerPosition.y > parameters.mapsize.y;

            if (done) {
                return -10;
            }

            var targetDist = (playerPosition - targetPosition).sqrMagnitude;
            var minDist = parameters.playerRadius + parameters.targetRadius;
            if (targetDist < minDist * minDist) {
                SetTargetPosition();
                return parameters.targetScore;
            }
            var dir = (targetPosition - playerPosition).normalized;
            return -dir.sqrMagnitude;
        }


        public void ToTensor(Tensor<float> result) {
            // 4 per rock, 5 rocks => 20
            // player pos => 1

            var dir = (targetPosition - playerPosition).normalized;
            result[0] = dir.x;
            result[1] = dir.y;
        }
        void SetTargetPosition() {
            var p1 = GetRandomPosition();
            var p2 = GetRandomPosition();

            var d1 = (p1 - playerPosition).sqrMagnitude;
            var d2 = (p2 - playerPosition).sqrMagnitude;

            var result = p1;
            var d = d1;

            if (d2 > d) {
                result = p2;
                d = d2;
            }
            
            targetPosition = result;
            targetPosition = new Vector2();
        }

        Vector2 GetRandomPosition() {
            return new Vector2(UnityEngine.Random.Range(0, parameters.mapsize.x), UnityEngine.Random.Range(0, parameters.mapsize.y));
        }
    }
}