using System.Collections.Generic;

using Unity.Profiling;


namespace DumbML {
    public static class ProfileUtility {
        static Dictionary<string, ProfilerMarker> _dict = new Dictionary<string, ProfilerMarker>();

        public static void Start(string s) {
            if (!_dict.ContainsKey(s)) {
                _dict.Add(s, new ProfilerMarker(s));
            }
            _dict[s].Begin();
        }

        public static void End(string s) {
            _dict[s].End();

        }
    }
}

