package category;

import javafx.util.Pair;

import java.util.*;

public class DFS {

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

    //    剑指 Offer 12. 矩阵中的路径 剪枝
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


    public static void main(String[] args) {
        DFS o = new DFS();
//        o.solveNQueens(4);

        char[][] c = new char[3][4];
        c[0] = new char[]{'A', 'B', 'C', 'E'};
        c[1] = new char[]{'S', 'F', 'C', 'S'};
        c[2] = new char[]{'A', 'D', 'E', 'E'};

        System.out.println(o.exist(c, "SEE"));
    }

}
