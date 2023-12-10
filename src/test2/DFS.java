package test2;

import javafx.util.Pair;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DFS {
    //    //698. 划分为k个相等的子集
////    List<List<Integer>> res = new ArrayList<>(20);
//    Map<Integer, Boolean> memo = new HashMap<>();
//
//    public boolean canPartitionKSubsets(int[] nums, int k) {
//        int sum = Arrays.stream(nums).sum();
//        if (sum % k != 0) {
//            return false;
//        }
//        int target = sum / k;
//        return backtrack(k, 0, target, nums, 0, 0);
//    }
//
//    public boolean backtrack(int k, int bucket, int target, int[] nums, int ind, int v) {
//        if (k == 0) {
//            return true;
//        }
//        if (bucket == target) {
//            boolean t = backtrack(k - 1, 0, target, nums, 0, v);
//            memo.put(v, t);
//            return t;
//        }
//        if (memo.containsKey(v)) {
//            return memo.get(v);
//        }
//        for (int i = ind; i < nums.length; i++) {
//            if (((v >> i) & 1) == 1) {
//                continue;
//            }
//            if (bucket + nums[i] > target) {
//                continue;
//            }
//
//            v |= (1 << i);
//            bucket += nums[i];
//            if (backtrack(k, bucket, target, nums, i + 1, v)) {
//                return true;
//            }
//            bucket -= nums[i];
//            v ^= (1 << i);
//        }
//        return false;
//    }
//
//
//    //    22. 括号生成
//    public List<String> generateParenthesis(int n) {
//        return null;
//    }
//
    //    51. N 皇后 回溯
    public List<List<String>> solveNQueens(int n) {
        int[][] nums = new int[n][n];
        List<List<String>> res = new ArrayList<>();
        backtrack(nums, 0, res, new LinkedList<>());
        return res;
    }

    public void backtrack(int[][] nums, int row, List<List<String>> res, Deque<Pair<Integer, Integer>> ind) {
        if (row == nums.length) {
            res.add(int2string(nums));
            return;
        }
        for (int i = 0; i < nums[row].length; i++) {
            if (valid(row, i, ind)) {
                nums[row][i] = 1;
                ind.offerLast(new Pair<>(row, i));
                backtrack(nums, row + 1, res, ind);
                nums[row][i] = 0;
                ind.pollLast();
            }
        }
    }

    public boolean valid(int i, int j, Deque<Pair<Integer, Integer>> ind) {
        return ind.stream().noneMatch(p -> p.getKey() == i || p.getValue() == j || Math.abs(p.getKey() - i) == Math.abs(p.getValue() - j));
    }

    public List<String> int2string(int[][] nums) {
        List<String> res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            StringBuilder sb = new StringBuilder();
            Arrays.stream(nums[i]).forEach(e -> sb.append(e == 0 ? '.' : 'Q'));
            res.add(sb.toString());
        }
        System.out.println(res);
        return res;
    }

    //    52. N 皇后 II
    public int totalNQueens(int n) {
        return solveNQueens(n).size();
    }


    //
//    //    剑指 Offer 12. 矩阵中的路径 79. 单词搜索  剪枝
//    public boolean exist(char[][] board, String word) {
////        上
//        move.add(new Pair<>(-1, 0));
////        下
//        move.add(new Pair<>(1, 0));
////        左
//        move.add(new Pair<>(0, -1));
////        右
//        move.add(new Pair<>(0, 1));
//
//        System.out.println(move);
//
//        for (int i = 0; i < board.length; i++) {
//            for (int j = 0; j < board[i].length; j++) {
//                if (dfs(board, i, j, word, 0, 0)) {
//                    return true;
//                }
//            }
//        }
//        return false;
//    }
//
//    List<Pair<Integer, Integer>> move = new ArrayList<>();
//
//    public boolean dfs(char[][] board, int i, int j, String word, int k, int n) {
////        MyUtile.printBlank(n++, "in i=" + i + ",j=" + j);
//        if (i < 0 || i > board.length - 1 || j < 0 || j > board[i].length - 1 || word.charAt(k) != board[i][j]) {
////            System.out.println("k="+word.charAt(k));
//            return false;
//        }
//        if (k == word.length() - 1) return true;
//
//        board[i][j] = '\0';
//        int finalN = n;
//        boolean nextRes = move.stream().anyMatch(m -> dfs(board, i + m.getKey(), j + m.getValue(), word, k + 1, finalN));
//        board[i][j] = word.charAt(k);
////        MyUtile.printBlank(--n, "out i=" + i + ",j=" + j + nextRes);
//        return nextRes;
//    }
//
////    public int movingCount(int m, int n, int k) {
////
////    }
////    public int bitNumCount(int n){
////
////    }
//
//
//    //78. 子集   不可复选回溯  90. 子集 II 结果去重 同框架
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        backtrackNoRepeat(nums, 0, new LinkedList(), res);
        System.out.println(res);
        return res;
    }

    public void backtrackNoRepeat(int[] nums, int start, Deque q, List<List<Integer>> res) {
//        if (start == nums.length) {
        res.add(new ArrayList<>(q));
//        }
        int pre = 99999;
        for (int i = start; i < nums.length; i++) {
            if (nums[i] != pre) {
                q.offerLast(pre = nums[i]);
                backtrackNoRepeat(nums, i + 1, q, res);
                q.pollLast();
            }
        }
    }

    //46. 全排列  47. 全排列 II
    public List<List<Integer>> permute(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        allPermute2(nums, new boolean[nums.length], new LinkedList<>(), res);
        return res;
    }

    public void allPermute(int[] nums, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        if (q.size() == nums.length) {
            res.add(new ArrayList<>(q));
        }

        for (int i = 0; i < nums.length; i++) {
            if (v[i]) {
                continue;
            }
//            剪枝的思路就是在一开始第一次选的时候相同的数要么都连续用上  要么都不用，所以条件就是对于连续相同的一片数字前面一个如果用上了的话我就必须用，反之就是如果前面不用那我也坚决不用
//            因此条件就是nums[i] == nums[i - 1] && !v[i - 1]的时候continue，例如2 2 2 2 2 3,结果要么是222223要么是322222 就是2必须在一块共同进退
//            对于2 2 2 3 2 2 的情况会在回溯的时候222322的下一个会变成->222232->222223这样全部的情况都有了
            if (i > 0 && nums[i] == nums[i - 1] && !v[i - 1]) {
                continue;
            }
            v[i] = !v[i];
            q.offerLast(nums[i]);
            allPermute(nums, v, q, res);
            v[i] = !v[i];
            q.pollLast();
        }
    }

    public void allPermute2(int[] nums, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        if (q.size() == nums.length) {
            res.add(new ArrayList<>(q));
        }
        int pre = 99999;
        for (int i = 0; i < nums.length; i++) {
//            剪枝的方式更好理解，就是当前准备要用的数和前一个用过了的数一样的话就不用，因为如果用了的话会导致后面的树枝都是一样的
            if (!v[i] && nums[i] != pre) {
                v[i] = !v[i];
                q.offerLast(pre = nums[i]);
                allPermute2(nums, v, q, res);
                v[i] = !v[i];
                q.pollLast();

            }
        }
    }

    //    全排列（元素无重可复选）
    public List<List<Integer>> allpermute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        allpermute(nums, new LinkedList<>(), res);
        return res;
    }

    public void allpermute(int[] nums, Deque<Integer> q, List<List<Integer>> res) {
        if (q.size() == nums.length) {
            res.add(new ArrayList<>(q));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            q.offerLast(nums[i]);
            allpermute(nums, q, res);
            q.pollLast();
        }
    }


    public static void swap(int[] nums, int a, int b) {
        int t = nums[a];
        nums[a] = nums[b];
        nums[b] = t;
    }


    //    //77. 组合
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        combine(IntStream.range(1, n + 1).toArray(), 0, k, new LinkedList<>(), res);
        return res;
    }

    public void combine(int[] nums, int start, int k, Deque<Integer> q, List<List<Integer>> res) {
        if (q.size() == k) {
            res.add(new ArrayList<>(q));
        }
        for (int i = start; i < nums.length; i++) {
            q.offerLast(nums[i]);
            combine(nums, i + 1, k, q, res);
            q.pollLast();
        }
    }

    //    //39. 组合总和   40. 组合总和 II https://labuladong.github.io/algo/di-san-zha-24031/bao-li-sou-96f79/hui-su-sua-56e11/
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSum(candidates, 0, target, new LinkedList<>(), res);
        return res;
    }

    public void combinationSum(int[] nums, int strart, int target, Deque<Integer> q, List<List<Integer>> res) {
        int sum = q.stream().mapToInt(Integer::intValue).sum();
        if (sum > target) {
            return;
        }
        if (sum == target) {
            res.add(new ArrayList<>(q));
            return;
        }
        for (int i = strart; i < nums.length; i++) {
            q.offerLast(nums[i]);
            combinationSum(nums, i, target, q, res);
            q.pollLast();
        }

    }

    //
//    //216. 组合总和 III
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        combinationSum3(IntStream.range(1, 10).toArray(), k, n, 0, 0, new HashMap<>(), new LinkedList<>(), res);
        return res;
    }

    public void combinationSum3(int[] nums, int k, int target, int strat, int v, Map<Integer, Boolean> memo, Deque<Integer> q, List<List<Integer>> res) {

        if (memo.containsKey(v)) {
            return;
        } else {
            memo.put(v, true);
        }
        int sum = q.stream().mapToInt(Integer::intValue).sum();
        if (sum > target) {
            return;
        }
        if (q.size() == k && sum == target) {
            res.add(new ArrayList<>(q));
            return;
        }
        for (int i = strat; i < nums.length; i++) {
            if (((v >> i) & 1) == 1) {
                continue;
            }
            v |= (1 << i);
            q.offerLast(nums[i]);
            combinationSum3(nums, k, target, strat + 1, v, memo, q, res);
            v ^= (1 << i);
            q.pollLast();
        }
    }

    //    //    200. 岛屿数量
    public int numIslands(char[][] grid) {
        List<Pair<Integer, Integer>> move = new ArrayList<>();
        move.add(new Pair<>(-1, 0));
        move.add(new Pair<>(1, 0));
        move.add(new Pair<>(0, -1));
        move.add(new Pair<>(0, 1));
        int sum = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int i1 = 0; i1 < grid[i].length; i1++) {
                if (grid[i][i1] == '1') {
                    sum++;
                    Set<Pair<Integer, Integer>> trace = new HashSet<>();
                    numIslands(grid, i, i1, move, trace);
                    trace.forEach(p -> grid[p.getKey()][p.getValue()] = '0');
                }
            }
        }
        return sum;
    }

    public void numIslands(char[][] grid, int i, int j, List<Pair<Integer, Integer>> move, Set<Pair<Integer, Integer>> trace) {
        if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == '1' && trace.add(new Pair<>(i, j))) {
            move.forEach(p -> numIslands(grid, i + p.getKey(), j + p.getValue(), move, trace));
        }
    }

    //    //    1254. 统计封闭岛屿的数目 是封闭的置为1
    public int closedIsland(int[][] grid) {
        List<Pair<Integer, Integer>> move = new ArrayList<>();
        move.add(new Pair<>(-1, 0));
        move.add(new Pair<>(1, 0));
        move.add(new Pair<>(0, -1));
        move.add(new Pair<>(0, 1));
        int sum = 0, m = grid.length, n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int i1 = 0; i1 < n; i1++) {
                if ((i == 0 || i == m - 1 || i1 == 0 || i1 == n - 1) && grid[i][i1] == 0) {
                    Set<Pair<Integer, Integer>> trace = new HashSet<>();
                    closedIsland(grid, i, i1, move, trace);
                    trace.forEach(p -> grid[p.getKey()][p.getValue()] = 1);
                }
            }
        }
        for (int i = 0; i < grid.length; i++) {
            for (int i1 = 0; i1 < grid[i].length; i1++) {
                System.out.print(grid[i][i1] + " ");
            }
            System.out.println();
        }
        for (int i = 0; i < grid.length; i++) {
            for (int i1 = 0; i1 < grid[i].length; i1++) {
                if (grid[i][i1] == 0) {
                    Set<Pair<Integer, Integer>> trace = new HashSet<>();
                    closedIsland(grid, i, i1, move, trace);
                    trace.forEach(p -> grid[p.getKey()][p.getValue()] = 1);
                    sum++;
                }
            }
        }
        return sum;
    }

    public void closedIsland(int[][] grid, int i, int j, List<Pair<Integer, Integer>> move, Set<Pair<Integer, Integer>> trace) {
        if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == 0 && trace.add(new Pair<>(i, j))) {
            move.forEach(p -> closedIsland(grid, i + p.getKey(), j + p.getValue(), move, trace));
        }
    }

    //
//
//    //130. 被围绕的区域 https://leetcode.cn/problems/surrounded-regions/
    public void solve(char[][] board) {
        List<Pair<Integer, Integer>> move = new ArrayList<>();
        move.add(new Pair<>(-1, 0));
        move.add(new Pair<>(1, 0));
        move.add(new Pair<>(0, -1));
        move.add(new Pair<>(0, 1));
        int sum = 0, m = board.length, n = board[0].length;
        Set<Pair<Integer, Integer>> trace = new HashSet<>();

        for (int i = 0; i < m; i++) {
            for (int i1 = 0; i1 < n; i1++) {
                if ((i == 0 || i == m - 1 || i1 == 0 || i1 == n - 1) && board[i][i1] == 'O') {
                    solve(board, i, i1, move, trace);
                    trace.forEach(p -> board[p.getKey()][p.getValue()] = 'X');
                }
            }
        }
        for (int i = 0; i < m; i++) {
            Arrays.fill(board[i], 'X');
        }
        trace.forEach(p -> board[p.getKey()][p.getValue()] = 'O');


        for (int i = 0; i < board.length; i++) {
            for (int i1 = 0; i1 < board[i].length; i1++) {
                System.out.print(board[i][i1] + " ");
            }
            System.out.println();
        }
    }

    public void solve(char[][] board, int i, int j, List<Pair<Integer, Integer>> move, Set<Pair<Integer, Integer>> trace) {
        if (i >= 0 && i < board.length && j >= 0 && j < board[0].length && board[i][j] == 'O' && !trace.contains(new Pair<>(i, j))) {
            trace.add(new Pair<>(i, j));
            move.forEach(p -> solve(board, i + p.getKey(), j + p.getValue(), move, trace));
        }
    }

    //
//    //695. 岛屿的最大面积
    public int maxAreaOfIsland(int[][] grid) {
        List<Pair<Integer, Integer>> move = new ArrayList<>();
        move.add(new Pair<>(1, 0));
        move.add(new Pair<>(-1, 0));
        move.add(new Pair<>(0, 1));
        move.add(new Pair<>(0, -1));

        int res = 0, m = grid.length, n = grid[0].length;
        Set<Pair<Integer, Integer>> trace = new HashSet<>();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                trace.clear();
                maxAreaOfIsland(grid, i, j, move, trace);
                res = Math.max(res, trace.size());
                trace.forEach(p -> grid[p.getKey()][p.getValue()] = 0);
            }
        }
        return res;

    }

    public void maxAreaOfIsland(int[][] grid, int i, int j, List<Pair<Integer, Integer>> move, Set<Pair<Integer, Integer>> trace) {
        if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == 1 && trace.add(new Pair<>(i, j))) {
            move.forEach(p -> maxAreaOfIsland(grid, i + p.getKey(), j + p.getValue(), move, trace));
        }
    }

    //
    public static void main(String[] args) {

        int[] nums = new int[]{1, 2, 3};
        DFS o = new DFS();
//        System.out.println(o.combinationSum3(3, 9));
//        System.out.println(o.solveNQueens(4));

        char[][] c = new char[4][4];
        c[0] = new char[]{'X', 'X', 'X', 'X'};
        c[1] = new char[]{'X', 'O', 'O', 'X'};
        c[2] = new char[]{'X', 'X', 'O', 'X'};
        c[3] = new char[]{'X', 'O', 'X', 'X'};
//        c[4] = new char[]{'X', 'X', 'X', 'X'};
//        c[1] = new char[]{'S', 'F', 'C', 'S'};
//        c[2] = new char[]{'A', 'D', 'E', 'E'};
//        c[0] = new char[]{'A', 'B', 'C', 'E'};
//        c[1] = new char[]{'S', 'F', 'C', 'S'};
//        c[2] = new char[]{'A', 'D', 'E', 'E'};
//        o.solve(c);
        System.out.println();

//        System.out.println(o.exist(c, "SEE"));
//        int[] nums = new int[]{2, 3, 6, 7};
//        o.combinationSum(nums, 7);

        int[][] grid = new int[5][8];
//        grid[0] = new int[]{0,0,1,0,0};
//        grid[1] = new int[]{0,1,0,1,0};
//        grid[2] = new int[]{0,1,1,1,0};

        grid[0] = new int[]{1, 1, 1, 1, 1, 1, 1, 0};
        grid[1] = new int[]{1, 0, 0, 0, 0, 1, 1, 0};
        grid[2] = new int[]{1, 0, 1, 0, 1, 1, 1, 0};
        grid[3] = new int[]{1, 0, 0, 0, 0, 1, 0, 1};
        grid[4] = new int[]{1, 1, 1, 1, 1, 1, 1, 0};
//        grid[5] = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
//        grid[6] = new int[]{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0};
//        grid[7] = new int[]{0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0};
//        grid[8] = new int[]{0, 0, 0, 1, 1, 0, 1, 1, 1, 0};
//        grid[9] = new int[]{1, 1, 0, 1, 0, 1, 0, 0, 1, 0};

        System.out.println(o.closedIsland(grid));
//        o.canPartitionKSubsets(new int[]{4, 3, 2, 3, 5, 2, 1}, 4);
    }
//
//
}
