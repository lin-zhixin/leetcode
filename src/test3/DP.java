package test3;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

public class DP {
    //198. 打家劫舍
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int n = nums.length;
        int[] dp = new int[n];
//        Arrays.fill(dp, 99999);
        dp[0] = nums[0];
        dp[1] = nums[1];
        int max = Math.max(dp[0], dp[1]);
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                dp[i] = Math.max(dp[i], dp[j] + nums[i]);
                max = Math.max(max, dp[i]);
            }

        }
//        System.out.println(Arrays.stream(dp).boxed().collect(Collectors.toList()));
        return max;


    }

    public boolean canJump(int[] nums) {

        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 99999);
        dp[0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (i - j <= nums[j]) {
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
//        System.out.println(Arrays.stream(dp).boxed().collect(Collectors.toList()));
        return dp[n - 1] != 99999;


    }

    //    97. 交错字符串
    public boolean isInterleave(String s1, String s2, String s3) {
        int[][][] memo = new int[s1.length()][s2.length()][s3.length()];
        for (int i = 0; i < s1.length(); i++) {
            for (int j = 0; j < s2.length(); j++) {
                Arrays.fill(memo[i][j], -1);
            }
        }
        return isInterleave(s1, s2, s3, 0, 0, 0, memo);
    }

    public boolean isInterleave(String s1, String s2, String s3, int i, int j, int k, int[][][] memo) {
        if (k == s3.length()) {
            return i == s1.length() && j == s2.length();
        }
        if (i == s1.length()) {
            return Objects.equals(s2.substring(j), s3.substring(k));
        }
        if (j == s2.length()) {
            return Objects.equals(s1.substring(i), s3.substring(k));
        }
        if (memo[i][j][k] != -1) {
            return memo[i][j][k] == 1;
        }

        char s1c = s1.charAt(i), s2c = s2.charAt(j), s3c = s3.charAt(k);
        if (s1c == s3c && isInterleave(s1, s2, s3, i + 1, j, k + 1, memo) || s2c == s3c && isInterleave(s1, s2, s3, i, j + 1, k + 1, memo)) {
            return 1 == (memo[i][j][k] = 1);
        }
        return 1 == (memo[i][j][k] = 0);
    }

    //1143. 最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
//                if ()
                dp[i][j] = Objects.equals(text1.charAt(i - 1), text2.charAt(j - 1)) ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[m][n];
    }

    public int longestCommonSubsequence(String text1, String text2, int i, int j, int[][] memo) {
        if (i == -1 || j == -1) {
            return 0;
        }
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        if (Objects.equals(text1.charAt(i), text2.charAt(j))) {
            return memo[i][j] = longestCommonSubsequence(text1, text2, i - 1, j - 1, memo) + 1;
        }
        return memo[i][j] = Math.max(longestCommonSubsequence(text1, text2, i - 1, j, memo), longestCommonSubsequence(text1, text2, i, j - 1, memo));
    }
}
