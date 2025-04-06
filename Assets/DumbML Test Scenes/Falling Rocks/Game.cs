using System.Collections.Generic;
using UnityEngine;


namespace FallingRocks {
    public class Game {
        public GameSettings settings { get; private set; }

        public float playerPos { get; private set; }
        public float spawnTimer { get; private set; }

        public List<RockInfo> rocks { get; private set; }


        public int numRocks => rocks.Count;
        public bool done;

        public Game(GameSettings gs) {
            settings = gs;
            playerPos = settings.width / 2;
            rocks = new List<RockInfo>();
        }

        public GameStatus Update(PlayerAction action) {
            UpdatePlayer(action);
            SpawnRocks();
            UpdateRocks();
            bool lose = CheckPlayerRockCollision();

            if (lose) {
                done = true;
                return GameStatus.Lose();
            }

            return GameStatus.KeepGoing();
        }

        public void Reset() {
            playerPos = settings.width / 2;
            rocks.Clear();
            done = false;
        }

        void UpdatePlayer(PlayerAction action) {
            float diff = 0;
            if (action == PlayerAction.left) {
                diff = -1;
            }
            else if (action == PlayerAction.right) {
                diff = 1;
            }

            playerPos += diff * settings.deltaTime *settings.playerSpeed;

            if (playerPos < 0) {
                playerPos = 0;
            }
            if (playerPos > settings.width) {
                playerPos = settings.width;
            }
        }

        void SpawnRocks() {
            spawnTimer += settings.deltaTime;
            if (spawnTimer < settings.rockSpawnInterval) {
                return;
            }

            if (numRocks >= settings.maxRocks) {
                return;
            }

            spawnTimer = 0;

            RockInfo rock = new RockInfo();
            rock.x = Random.Range(0, settings.width);
            rock.y = settings.height;
            rock.radius = Random.Range(settings.rockRadiusMin, settings.rockRadiusMax);

            rock.dx = Random.Range(-.1f, .1f);
            rocks.Add(rock);
        }

        void UpdateRocks() {
            for (int i = 0; i < rocks.Count; i++) {
                RockInfo rock = rocks[i];

                rock.y -= settings.rockSpeed * settings.deltaTime;
                rock.x += settings.rockSpeed * settings.deltaTime * rock.dx;

                rocks[i] = rock;
            }

            // remove finished rocks
            for (int i = rocks.Count - 1; i >= 0; i--) {
                RockInfo rock = rocks[i];

                bool remove = false;

                if (rock.y <= 0) {
                    remove = true;
                }
                if (rock.x <= 0) {
                    remove = true;
                }
                if (rock.x >= settings.width) {
                    remove = true;
                }

                if (remove) {
                    rocks.RemoveAt(i);
                }
            }
        }

        bool CheckPlayerRockCollision() {
            foreach (var r in rocks) {
                float dd = new Vector2(r.x - playerPos, r.y).sqrMagnitude;

                float dmin = (settings.playerRadius + r.radius) * (settings.playerRadius + r.radius);

                if (dd <= dmin) {
                    return true;
                }
            }
            
            return false;
        }
    }
}
