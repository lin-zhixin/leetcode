package category;

import javafx.util.Pair;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

public class DFS {
    //698. 划分为k个相等的子集
//    List<List<Integer>> res = new ArrayList<>(20);
    Map<Integer, Boolean> memo = new HashMap<>();

    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = Arrays.stream(nums).sum();
        if (sum % k != 0) {
            return false;
        }
        int target = sum / k;
        return backtrack(k, 0, target, nums, 0, 0);
    }

    public boolean backtrack(int k, int bucket, int target, int[] nums, int ind, int v) {
        if (k == 0) {
            return true;
        }
        if (bucket == target) {
            boolean t = backtrack(k - 1, 0, target, nums, 0, v);
            memo.put(v, t);
            return t;
        }
        if (memo.containsKey(v)) {
            return memo.get(v);
        }
        for (int i = ind; i < nums.length; i++) {
            if (((v >> i) & 1) == 1) {
                continue;
            }
            if (bucket + nums[i] > target) {
                continue;
            }

            v |= (1 << i);
            bucket += nums[i];
            if (backtrack(k, bucket, target, nums, i + 1, v)) {
                return true;
            }
            bucket -= nums[i];
            v ^= (1 << i);
        }
        return false;
    }


    //    22. 括号生成
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        generateParenthesis(n, 0, 0, new StringBuilder(), res);
        return res;
    }

    public void generateParenthesis(int n, int l, int r, StringBuilder sb, List<String> res) {
        if (sb.length() == 2 * n) {
            res.add(sb.toString());
            return;
        }
        //            如果左括号的个数小于总数的一半就继续塞进去
        if (l < n) {
            sb.append('(');
            generateParenthesis(n, l + 1, r, sb, res);
            sb.deleteCharAt(sb.length() - 1);
        }
        //            如果右括号的数量小于左边的就开始赛右括号
        if (r < l) {
            sb.append(')');
            generateParenthesis(n, l, r + 1, sb, res);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    //    51. N 皇后 回溯
    public List<List<String>> solveNQueens(int n) {
        int[][] map = new int[n][n];
        List<List<String>> res = new ArrayList<>();
        backTrace(map, 0, res, new LinkedList<>());
        System.out.println(res);
        return res;
    }

    public void backTrace(int[][] map, int row, List<List<String>> res, Deque<Pair<Integer, Integer>> index) {
        if (row >= map.length) {
            res.add(map2String(map));
            return;
        }
        for (int i = 0; i < map[row].length; i++) {
            if (isvalid(row, i, index)) {
                index.addLast(new Pair<>(row, i));
                map[row][i] = 1;
                backTrace(map, row + 1, res, index);
                map[row][i] = 0;
                index.removeLast();
            }
        }
    }

    public boolean isvalid(int row, int col, Deque<Pair<Integer, Integer>> index) {
        return index.stream().noneMatch(ind -> (ind.getValue() == col || Math.abs(ind.getKey() - row) == Math.abs(ind.getValue() - col)));
    }

    public List<String> map2String(int[][] map) {
        List<String> res = new ArrayList<>();
        for (int i = 0; i < map.length; i++) {
            StringBuilder tmp = new StringBuilder();
            for (int j = 0; j < map[i].length; j++) {
                tmp.append(map[i][j] == 0 ? "." : "Q");
            }
            res.add(tmp.toString());
        }
        return res;
    }

    //    剑指 Offer 12. 矩阵中的路径 79. 单词搜索  剪枝
    public boolean exist(char[][] board, String word) {
//        上
        move.add(new Pair<>(-1, 0));
//        下
        move.add(new Pair<>(1, 0));
//        左
        move.add(new Pair<>(0, -1));
//        右
        move.add(new Pair<>(0, 1));

        System.out.println(move);

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (dfs(board, i, j, word, 0, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    List<Pair<Integer, Integer>> move = new ArrayList<>();

    public boolean dfs(char[][] board, int i, int j, String word, int k, int n) {
//        MyUtile.printBlank(n++, "in i=" + i + ",j=" + j);
        if (i < 0 || i > board.length - 1 || j < 0 || j > board[i].length - 1 || word.charAt(k) != board[i][j]) {
//            System.out.println("k="+word.charAt(k));
            return false;
        }
        if (k == word.length() - 1) return true;

        board[i][j] = '\0';
        int finalN = n;
        boolean nextRes = move.stream().anyMatch(m -> dfs(board, i + m.getKey(), j + m.getValue(), word, k + 1, finalN));
        board[i][j] = word.charAt(k);
//        MyUtile.printBlank(--n, "out i=" + i + ",j=" + j + nextRes);
        return nextRes;
    }

//    public int movingCount(int m, int n, int k) {
//
//    }
//    public int bitNumCount(int n){
//
//    }


    //78. 子集   不可复选回溯  90. 子集 II 结果去重 同框架
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();

        backtrackNoRepeat(nums, 0, new LinkedList<>(), res);
        System.out.println(res);
        return res;
    }

    public void backtrackNoRepeat(int[] nums, int start, Deque<Integer> trace, List<List<Integer>> res) {
        List<Integer> t = new ArrayList<>(trace).stream().sorted().collect(Collectors.toList());
        if (!res.contains(t) && res.add(t)) ;
        for (int i = start; i < nums.length; i++) {
            trace.addLast(nums[i]);
            backtrackNoRepeat(nums, i + 1, trace, res);
            trace.removeLast();
        }
    }


    //46. 全排列  47. 全排列 II
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        allPermute(nums, nums.length, 0, new LinkedList<>(), res);
        System.out.println(res);
        return res;
    }

    public void allPermute(int[] nums, int k, int start, Deque<Integer> trace, List<List<Integer>> res) {
//        k代表截取二叉树的第几层的结果
        if (trace.size() >= k) {
            List<Integer> t = new ArrayList<>(trace).stream().sorted().collect(Collectors.toList());
            if (!res.contains(t) && res.add(t)) ;
            return;
        }

        for (int i = start; i < nums.length; i++) {
            trace.addLast(nums[i]);
//            if (nums[start] != nums[i]) {
//                nums[start] = nums[start] ^ nums[i];
//                nums[i] = nums[start] ^ nums[i];
//                nums[start] = nums[start] ^ nums[i];
//            }
//            不可复选使用i+1   可复选使用start+1
            allPermute(nums, k, i + 1, trace, res);
//            if (nums[start] != nums[i]) {
//                nums[start] = nums[start] ^ nums[i];
//                nums[i] = nums[start] ^ nums[i];
//                nums[start] = nums[start] ^ nums[i];
//            }
            trace.removeLast();
        }
    }

    //77. 组合
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        int[] nums = new int[n];
        for (int i = 1; i <= n; i++) {
            nums[i - 1] = i;
        }
        allPermute(nums, k, 0, new LinkedList<>(), res);
        System.out.println(res);
        return res;

    }

    //39. 组合总和   40. 组合总和 II https://labuladong.github.io/algo/di-san-zha-24031/bao-li-sou-96f79/hui-su-sua-56e11/
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        combineTarget(candidates, 0, target, new LinkedList<>(), res);
        System.out.println(res);
        return res;
    }

    public void combineTarget(int[] nums, int start, int target, Deque<Integer> trace, List<List<Integer>> res) {
        int sum = trace.stream().mapToInt(Integer::intValue).sum();
        if (sum >= target) {
            if (sum == target && res.add(new ArrayList<>(trace))) ;
            return;
        }

        for (int i = start; i < nums.length; i++) {
//            剪枝去重
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            trace.addLast(nums[i]);
            combineTarget(nums, i, target, trace, res);
            trace.removeLast();
        }
    }

    //216. 组合总和 III
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        int[] nums = new int[9];
        for (int i = 1; i <= 9; i++) {
            nums[i - 1] = i;
        }

        Arrays.sort(nums);
        combineTarget2(nums, 0, n, k, new LinkedList<>(), res);
        System.out.println(res);
        return res;


    }

    public void combineTarget2(int[] nums, int start, int target, int k, Deque<Integer> trace, List<List<Integer>> res) {
        if (trace.size() > k) {
            return;
        }
        int sum = trace.stream().mapToInt(Integer::intValue).sum();
        if (sum >= target) {
            if (sum == target && trace.size() == k && res.add(new ArrayList<>(trace))) ;
            return;
        }

        for (int i = start; i < nums.length; i++) {
//            剪枝去重
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            trace.addLast(nums[i]);
            combineTarget2(nums, i + 1, k, target, trace, res);
            trace.removeLast();
        }
    }


    //    200. 岛屿数量
    public int numIslands(char[][] grid) {
//        上
        move.add(new Pair<>(-1, 0));
//        下
        move.add(new Pair<>(1, 0));
//        左
        move.add(new Pair<>(0, -1));
//        右
        move.add(new Pair<>(0, 1));

        System.out.println(move);

        int res = 0;
        int m = grid.length, n = grid[0].length;
        boolean[][] v = new boolean[m][n];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == '1') {
                    res++;
                    numIslandsDFS(grid, i, j, v);
                }
            }
        }
        System.out.println(res);
        return res;

    }

    public void numIslandsDFS(char[][] grid, int i, int j, boolean[][] v) {
        int m = grid.length, n = grid[0].length;
        if (i < 0 || i >= m || j < 0 || j >= n || v[i][j] || grid[i][j] == '0') {
            return;
        }
        v[i][j] = true;
        grid[i][j] = '1';
        move.forEach(mo -> {
            numIslandsDFS(grid, i + mo.getKey(), j + mo.getValue(), v);
        });
    }


    //    1254. 统计封闭岛屿的数目 是封闭的置为1
    public int closedIsland(int[][] grid) {
        move.add(new Pair<>(-1, 0));
        move.add(new Pair<>(1, 0));
        move.add(new Pair<>(0, -1));
        move.add(new Pair<>(0, 1));
        System.out.println(move);

        int res = 0;
        int m = grid.length, n = grid[0].length;
        boolean[][] v = new boolean[m][n];
        Set<Pair<Integer, Integer>> exInd = new HashSet<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 && !v[i][j]) {
                    if (closedIslandDFS(grid, i, j, v, exInd)) {
                        System.out.println(i + "," + j);
                        res++;
                        exInd.forEach(p -> grid[p.getKey()][p.getValue()] = 1);
                        exInd.clear();
                    }
                }
            }
        }
        System.out.println(res);
        return res;

    }

    public boolean closedIslandDFS(int[][] grid, int i, int j, boolean[][] v, Set<Pair<Integer, Integer>> exInd) {
        int m = grid.length, n = grid[0].length;
        if ((i == 0 || i == m - 1 || j == 0 || j == n - 1) && grid[i][j] == 0) {
            return false;
        }
        if (grid[i][j] == 1 || v[i][j]) {
            return true;
        }
        v[i][j] = true;
        exInd.add(new Pair<>(i, j));
        AtomicBoolean is = new AtomicBoolean(true);
        move.forEach(mo -> {
            if (!closedIslandDFS(grid, i + mo.getKey(), j + mo.getValue(), v, exInd)) {
                is.set(false);
            }
        });
        return is.get();
    }


    //130. 被围绕的区域 https://leetcode.cn/problems/surrounded-regions/
    public void solve(char[][] board) {
//        上
        move.add(new Pair<>(-1, 0));
//        下
        move.add(new Pair<>(1, 0));
//        左
        move.add(new Pair<>(0, -1));
//        右
        move.add(new Pair<>(0, 1));

        System.out.println(move);

        int m = board.length, n = board[0].length;
        boolean[][] v = new boolean[m][n];
        Set<Pair<Integer, Integer>> exInd = new HashSet<>();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if ((i == 0 || j == 0 || i == m - 1 || j == n - 1) && board[i][j] == 'O') {

                    solveDFS(board, i, j, v, exInd);
                }
            }
        }
        System.out.println(exInd);

        for (int i = 0; i < board.length; i++) {
            Arrays.fill(board[i], 'X');
        }
        exInd.forEach(p -> board[p.getKey()][p.getValue()] = 'O');


    }

    public void solveDFS(char[][] grid, int i, int j, boolean[][] v, Set<Pair<Integer, Integer>> exInd) {
        int m = grid.length, n = grid[0].length;
        if (i < 0 || i >= m || j < 0 || j >= n || v[i][j] || grid[i][j] == 'X') {
            return;
        }
        v[i][j] = true;
        exInd.add(new Pair<>(i, j));
        for (Pair<Integer, Integer> mo : move) {
            solveDFS(grid, i + mo.getKey(), j + mo.getValue(), v, exInd);
        }
    }

    //695. 岛屿的最大面积
    public int maxAreaOfIsland(int[][] grid) {
        move.add(new Pair<>(-1, 0));
        move.add(new Pair<>(1, 0));
        move.add(new Pair<>(0, -1));
        move.add(new Pair<>(0, 1));
        System.out.println(move);

        int res = 0;
        int m = grid.length, n = grid[0].length;
        boolean[][] v = new boolean[m][n];
        Set<Pair<Integer, Integer>> exInd = new HashSet<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    res = Math.max(res, maxAreaOfIslandDFS(grid, i, j, v));
                }
            }
        }
        return res;
    }

    public int maxAreaOfIslandDFS(int[][] grid, int i, int j, boolean[][] v) {
        int m = grid.length, n = grid[0].length;
        if (i < 0 || i >= m || j < 0 || j >= n || v[i][j] || grid[i][j] == 0) {
            return 0;
        }
        v[i][j] = true;
        int res = 1;
        for (Pair<Integer, Integer> mo : move) {
            res += maxAreaOfIslandDFS(grid, i + mo.getKey(), j + mo.getValue(), v);
        }
        return res;
    }


    public static void main(String[] args) {
        DFS o = new DFS();
//        o.solveNQueens(4);
        System.out.println(o.generateParenthesis(3));

        char[][] c = new char[3][4];
        c[0] = new char[]{'A', 'B', 'C', 'E'};
        c[1] = new char[]{'S', 'F', 'C', 'S'};
        c[2] = new char[]{'A', 'D', 'E', 'E'};


        System.out.println(o.combinationSum3(3, 7));
        int[] nums = new int[]{2, 3, 6, 7};
//        o.combinationSum(nums, 7);

        int[][] grid = new int[10][10];
        grid[0] = new int[]{1, 1, 0, 1, 1, 1, 1, 1, 1, 1};
        grid[1] = new int[]{0, 0, 1, 0, 0, 1, 0, 1, 1, 1};
        grid[2] = new int[]{1, 0, 1, 0, 0, 0, 1, 0, 1, 0};
        grid[3] = new int[]{1, 1, 1, 1, 1, 0, 0, 1, 0, 0};
        grid[4] = new int[]{1, 0, 1, 0, 1, 1, 1, 1, 1, 0};
        grid[5] = new int[]{0, 0, 0, 0, 1, 1, 0, 0, 0, 0};
        grid[6] = new int[]{1, 0, 1, 0, 0, 0, 0, 1, 1, 0};
        grid[7] = new int[]{1, 1, 0, 0, 1, 1, 0, 0, 0, 0};
        grid[8] = new int[]{0, 0, 0, 1, 1, 0, 1, 1, 1, 0};
        grid[9] = new int[]{1, 1, 0, 1, 0, 1, 0, 0, 1, 0};

//        o.closedIsland(grid);
//        o.canPartitionKSubsets(new int[]{4, 3, 2, 3, 5, 2, 1}, 4);
    }


}
