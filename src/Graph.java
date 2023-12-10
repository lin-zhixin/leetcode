import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class Graph {
    //797. 所有可能的路径 有向无环图的所有遍历
    List<List<Integer>> res;

    public List<List<Integer>> allPathsSourceTarget2(int[][] graph) {
        List<Integer>[] map = buildG(graph);
        v = new boolean[map.length];
        res = new ArrayList<>();
        dfs(map, 0, new LinkedList<>());
        return res;
    }

    public List<Integer>[] buildG(int[][] graph) {
        List<Integer>[] map = new ArrayList[graph.length];
        for (int i = 0; i < map.length; i++) {
            map[i] = new ArrayList<>();
        }

        for (int i = 0; i < graph.length; i++) {
            map[i].addAll(Arrays.stream(graph[i]).boxed().collect(Collectors.toList()));
        }
        return map;
    }

    public void dfs(List<Integer>[] map, int cur, Deque<Integer> path) {
//        if (v[cur]) {
//            return;
//        }
//        v[cur] = true;
        path.offerLast(cur);
        if (cur == map.length - 1) {
            res.add(new ArrayList<>(path));
        }
        map[cur].forEach(e -> dfs(map, e, path));
        path.pollLast();
    }


    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        int n = graph.length;
        traverse(graph, 0, new int[n][n], new ArrayDeque<>(), res);
        return res;
    }

    public void traverse(int[][] graph, int val, int[][] v, Deque<Integer> path, List<List<Integer>> res) {
        path.offerLast(val);
        int n = graph.length;
        if (val == n - 1) {
            res.add(new ArrayList<>(path));
            path.removeLast();
            return;
        }
        for (int i : graph[val]) {
            traverse(graph, i, v, path, res);
        }
        path.removeLast();
    }

    //  207. 课程表 拓扑排序
    boolean[] path;
    boolean[] v;
    boolean hashcycle = false;

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<Integer>[] g = buildGraph(numCourses, prerequisites);
        path = new boolean[numCourses];
        v = new boolean[numCourses];
        for (int i = 0; i < numCourses; i++) {
            hascycle(g, i);
        }
        return !hashcycle;
    }


    public void hascycle(List<Integer>[] g, int val) {
        if (path[val]) {
            hashcycle = true;
            return;
        }
        if (hashcycle || v[val]) {
            return;
        }
        v[val] = true;
        path[val] = true;
        for (Integer to : g[val]) {
            hascycle(g, to);
        }
        path[val] = false;
    }

    public List<Integer>[] buildGraph(int numCourses, int[][] prerequisites) {
        List<Integer>[] g = new ArrayList[numCourses];
        for (int i = 0; i < numCourses; i++) {
            g[i] = new ArrayList<>();
        }
        for (int[] prerequisite : prerequisites) {
            g[prerequisite[1]].add(prerequisite[0]);
        }
        return g;
    }

    //210. 课程表 II 拓扑排序（dfs实现 实际上就是多叉树后续遍历再反转）
    boolean hasCycles2 = false;
    boolean[] path2, v2;
    List<Integer> postOrder = new ArrayList<>();

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        path2 = new boolean[numCourses];
        v2 = new boolean[numCourses];
        List<Integer>[] g = buildG(numCourses, prerequisites);
        for (int i = 0; i < numCourses; i++) {
            topSort(g, i);
            if (hasCycles2) {
                return new int[0];
            }
        }
        Collections.reverse(postOrder);
        return postOrder.stream().mapToInt(Integer::intValue).toArray();
    }

    public List<Integer>[] buildG(int numCourses, int[][] prerequisites) {
        List<Integer>[] g = new ArrayList[numCourses];
        for (int i = 0; i < numCourses; i++) {
            g[i] = new ArrayList<>();
        }
        for (int[] edge : prerequisites) {
            g[edge[1]].add(edge[0]);
        }
        return g;
    }

    public void topSort(List<Integer>[] g, int s) {
        if (path2[s]) {
            hasCycles2 = true;
            return;
        }
        if (hasCycles2 || v2[s]) {
            return;
        }
        v2[s] = true;
        path2[s] = true;
        for (int to : g[s]) {
            topSort(g, to);
        }
        postOrder.add(s);
        path2[s] = false;
    }

    public static void main(String[] args) {
        Graph graph = new Graph();
        System.out.println(graph.findOrder(2, new int[0][0]));
    }
}
