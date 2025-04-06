namespace FallingRocks {
    public struct GameStatus {
        public bool done;
        public float reward;

        public static GameStatus Lose() {
            GameStatus result = new GameStatus();

            result.done = true;
            result.reward = -1;
            return result;
        }
        public static GameStatus KeepGoing() {
            GameStatus result = new GameStatus();

            result.done = false;
            result.reward = 1;

            return result;
        }
        public static GameStatus TimeLimit() {
            GameStatus result = new GameStatus();

            result.done = true;
            result.reward = 1;

            return result;
        }
    }
}
