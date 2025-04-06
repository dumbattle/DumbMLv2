namespace FallingRocks {
    public struct GameSettings {
        public float width;
        public float height;

        public float playerRadius;
        public float playerSpeed;

        public float rockSpeed;
        public float rockSpawnInterval;
        public float rockRadiusMin;
        public float rockRadiusMax;
        public float deltaTime;
        public int maxRocks;
        public static GameSettings Default() {
            GameSettings result = new GameSettings();

            result.width = 6;
            result.height = 8;

            result.playerRadius = .5f;
            result.playerSpeed = 1;

            result.rockSpeed = 4;
            result.rockSpawnInterval = 4.5f;
            result.rockRadiusMin = .25f;
            result.rockRadiusMax = .75f;
            result.deltaTime = 1 / 60f;
            result.maxRocks = 5;
            return result;
        }
    }

}
