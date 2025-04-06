using DumbML;


namespace FallingRocks {
    public static class GameUtility {
        public static int StateSize(this Game g) {
            return 32;
        }
        public static void ToTensor(this Game g, Tensor<float> result) {
            // 4 per rock, 5 rocks => 20
            // player pos => 1

            if (result == null) {
                result = new FloatTensor(1, g.StateSize());
            }
            for (int i = 1; i < g.StateSize(); i++) {
                float x = g.settings.width  * (i - 0.5f) / (g.StateSize() - 1);

                foreach (var rock in g.rocks) {
                    var min = rock.x - rock.radius;
                    var max = rock.x + rock.radius;
                    result[0, i] = x > min && x < max ? rock.y: g.settings.height;
                }
            }
          
            result[0, 0] = g.playerPos / g.settings.width;
        }
    }
}
