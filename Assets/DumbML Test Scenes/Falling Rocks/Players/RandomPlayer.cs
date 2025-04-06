using UnityEngine;


namespace FallingRocks {
    public class RandomPlayer : IPlayer {
        public PlayerAction GetAction(Game g) {
            int i = Random.Range(0, 3);
            return (PlayerAction)i;
        }
    }
}
