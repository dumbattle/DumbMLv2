using UnityEngine;
using LPE;
using System.Collections.Generic;



namespace FallingRocks {
    public class GameDisplayBehaviour : MonoBehaviour {
        public GameObject background;
        public GameObject player;
        public GameObject rockSrc;

        ObjectPool<GameObject> rockPool;
        List<GameObject> activeRocks = new List<GameObject>();


        private void Start() {
            rockPool = new ObjectPool<GameObject>(() => Instantiate(rockSrc));
        }
        public void DisplayGame(Game g) {
            // background
            background.transform.localScale = new Vector3(g.settings.width, g.settings.height, 1);
            background.transform.position = Vector3.zero;

            // player
            player.transform.localScale = new Vector3(g.settings.playerRadius, g.settings.playerRadius,.5f) * 2;
            player.transform.position = new Vector3(g.playerPos - g.settings.width / 2, -g.settings.height / 2);

            // rocks
            foreach (var r in activeRocks) {
                rockPool.Return(r);
                r.SetActive(false);
            }
            activeRocks.Clear();


            foreach (var r in g.rocks) {
                var rock = rockPool.Get();

                rock.transform.localScale = new Vector3(r.radius, r.radius, .5f) * 2;
                rock.transform.position = new Vector3(r.x - g.settings.width / 2, r.y - g.settings.height / 2);

                activeRocks.Add(rock);
                rock.SetActive(true);
            }
        }
    }

}
