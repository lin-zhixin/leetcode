package test2;

import category.MyUtile;
import category.SortTest;
import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class Dp {
    //    //70. 爬楼梯
    public int climbStairs(int n) {
        int[] res = new int[n + 1];
        res[0] = 1;
        res[1] = 1;
        for (int i = 2; i <= n; i++) {
            res[i] = res[i - 1] + res[i - 2];
        }
        return res[n];
    }

    //    //    300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int n = nums.length, res = 1;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
//        MyUtile.dis(dp);
        return res;
    }

    //    //53. 最大子数组和
    public int maxSubArray(int[] nums) {
        int n = nums.length, max = nums[0], pre = nums[0];
        for (int i = 1; i < n; i++) {
            pre = Math.max(nums[i], pre + nums[i]);
            max = Math.max(pre, max);
        }
        return max;
    }

    //    //1143. 最长公共子序列 备忘录+dp递归
////    https://labuladong.github.io/algo/di-er-zhan-a01c6/zi-xu-lie--6bc09/dong-tai-g-c481e/
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
//        int[][] dp = new int[m+1][n+1];
//        dp[i][j]=1+dp[i-1][j-1];
//        dp[i][j]=dp[i-1][j];
        int[][] memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            Arrays.fill(memo[i], -1);
        }
        return dp(text1, 0, text2, 0, memo);
    }

    public int dp(String text1, int i, String text2, int j, int[][] memo) {
        if (i == text1.length() || j == text2.length()) {
            return 0;
        }
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        if (text1.charAt(i) == text2.charAt(j)) {
            return memo[i][j] = dp(text1, i + 1, text2, j + 1, memo) + 1;
        }
        return memo[i][j] = Math.max(dp(text1, i + 1, text2, j, memo), dp(text1, i, text2, j + 1, memo));
    }

    //1143. 最长公共子序列 dp迭代做法
    public int longestCommonSubsequence2(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else {
//                    因为外循环i-1是已经算过了的也就是dp[i - 1][j]是算过了的，所以可以直接用，同理dp[i][j - 1]是算过了的，也可以直接用
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }

    //    516. 最长回文子序列
//    https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247486872&idx=1&sn=45b0cf438d3fb140ad6d06e660a7fa4d&chksm=9bd7f390aca07a86c37e440bb9d5fb936be0a1283c781cc2ad867f6e890b2e1ff26545a36158&mpshare=1&scene=1&srcid=0902o47FrATjXmehrLRI5tWM&sharer_sharetime=1599017372354&sharer_shareid=f21488698f08111d3eb92e4ab561e2fa&key=581a599b97d974ddfe4bc9650b2486f9e5201164cd7bf5512d4ff74880c52efff7b437525a6b0a296ad0d679517e35537727eac254b8d8949f232d1ffc5cc3a1a0b4db24a0f3842ad1db52029a050f77a1b69a9df6e619dc860c0e1cbce97f5bc92fe91128bbd9ca2036eb7f30aa4fa52985b46725f5aafd1303a99a5b23f08c&ascene=1&uin=MjA4NzM5MDAzMw%3D%3D&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=A3zj%2F8a0ytqk5A9Ek2EhUGA%3D&pass_ticket=DgOAivcZw7FjpzBz%2Bk27pvXXnDy9cjL9X1s2wX7FVNZ47Tk5byKdNegPXBIJG4ii
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    dp[i][j] = 1;
                    continue;
                }
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i + 1][j]);
                }
            }
        }
        return dp[0][n - 1];
    }

    //    5. 最长回文子串
    public String longestPalindrome(String s) {
        int n = s.length(), resi = 0, resj = 0, max = 0;
        boolean[][] dp = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    dp[i][j] = true;
                    continue;
                }
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = j - i == 1 || dp[i + 1][j - 1];
                }
                if ((j - i) > max && dp[i][j]) {
                    max = j - i;
                    resi = i;
                    resj = j;
                }
            }
        }
        MyUtile.disMap(dp);
        return s.substring(resi, resj + 1);
    }

    //1312. 让字符串成为回文串的最少插入次数
    public int minInsertions(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1];
                } else {
//                    步骤一：i~j-1如果已经是回文串那么在左边再插入一等于s[j]的字母就可以了，同理 如果i+1~j如果已经是回文串那么在右边再插入一等于s[j]的字母就可以了
//                    一种可以排除的情况就是i~j-1和i+1~j都不是回文，那么是先把其中某一个先变成回文，之后再按照步骤一，至于如何变不用管，相当于子问题，是i~j-1或者i+1~j的问题，直接取dp[i][j-1]或者dp[i+1][j]就行
                    dp[i][j] = Math.min(dp[i][j - 1], dp[i + 1][j]) + 1;
                }
            }
        }
        return dp[0][n - 1];

    }


    //
//
//    //583. 两个字符串的删除操作 求出最长公共字串之后各减去这个lcs长度
    public int minDistance4(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
//        MyUtile.disMap(dp);
        return m + n - 2 * dp[m][n];
    }


    //    //583. 两个字符串的删除操作 dp做法
    public int minDistance3(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            Arrays.fill(memo[i], -1);
        }
        return dp2(word1, 0, word2, 0, memo);
    }

    public int dp2(String word1, int i, String word2, int j, int[][] memo) {
        int m = word1.length(), n = word2.length();
        if (i == m) {
            return n - j;
        }
        if (j == n) {
            return m - i;
        }
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        if (word1.charAt(i) == word2.charAt(j)) {
            return memo[i][j] = dp2(word1, i + 1, word2, j + 1, memo);
        } else {
            return memo[i][j] = Math.min(dp2(word1, i + 1, word2, j, memo), dp2(word1, i, word2, j + 1, memo)) + 1;
        }
    }

    //
//    //55. 跳跃游戏
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 99999);
        dp[0] = 0;
//        dp[1] = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] >= i - j) {
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[n - 1] != 99999;
    }

    //    //45. 跳跃游戏 II
    public int jump(int[] nums) {

//        dp[i]:到当前位置最小次数
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 99999);
        dp[0] = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] >= i - j) {
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
        new category.SortTest().dis(dp);
        return dp[nums.length - 1];

    }

    ////62. 不同路径
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0) {
                    dp[i][j] = 1;
                    continue;
                }
                if (j == 0) {
                    dp[i][j] = 1;
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
//        MyUtile.disMap(dp);
        return dp[m - 1][n - 1];

    }

    //
//    //64. 最小路径和
    public int minPathSum(int[][] grid) {

        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = grid[i][0] + dp[i - 1][0];
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = grid[0][i] + dp[0][i - 1];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    //    //152. 乘积最大子数组
    public int maxProduct(int[] nums) {
        int n = nums.length;
        int[] mindp = new int[n];
        int[] maxdp = new int[n];
        mindp[0] = maxdp[0] = nums[0];
        for (int i = 1; i < n; i++) {
            mindp[i] = Math.min(Math.min(mindp[i - 1] * nums[i], maxdp[i - 1] * nums[i]), nums[i]);
            maxdp[i] = Math.max(Math.max(mindp[i - 1] * nums[i], maxdp[i - 1] * nums[i]), nums[i]);
        }
        return Arrays.stream(maxdp).max().orElse(0);
    }

    //
//    //221. 最大正方形
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];
        int max = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j] = matrix[i][j] == '0' ? 0 : 1;
                max = Math.max(max, dp[i][j]);
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (dp[i][j] == 1 && dp[i - 1][j] == 1 && dp[i][j - 1] == 1 && dp[i - 1][j - 1] == 1) {
                    dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
                }
                max = Math.max(max, dp[i][j]);
            }
        }
        disMap(dp);
        return max * max;
    }

    //
//    //416. 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = Arrays.stream(nums).sum(), n = nums.length;
        if (sum % 2 == 1) {
            return false;
        }
        sum /= 2;
        int[][] dp1 = new int[n + 1][sum + 1];
        int[] dp = new int[sum + 1];
        int[] pre = new int[sum + 1];
//        int pre = 0;
        for (int i = 1; i <= n; i++) {
//            pre = dp;
            for (int j = 0; j < sum + 1; j++) {
                int val = nums[i - 1];
                if (j < val) {
                    dp[j] = pre[j];
//                    dp1[i][j] = dp1[i - 1][j];
                } else {
                    dp[j] = Math.max(pre[j - val] + val, pre[j]);
//                    dp1[i][j] = Math.max(dp1[i - 1][j - val] + val, dp1[i - 1][j]);
                }
            }
            for (int j = 0; j < sum + 1; j++) {
                pre[j] = dp[j];
            }
        }
        MyUtile.dis(dp);
        System.out.println();
        MyUtile.disMap(dp1);
        return dp[sum] == sum;


    }

    //    //91. 解码方法 dp做法
    public int numDecodings2(String s) {
        int n = s.length();
        int[] dp = new int[n];
        dp[0] = s.charAt(0) == '0' ? 0 : 1;
//        dp[1] =
        for (int i = 1; i < n; i++) {
            dp[i] = s.charAt(i) == '0' ? 0 : dp[i - 1];
            int t = Integer.parseInt(s.substring(i - 1, i + 1));
            if (t > 0 && t <= 26 && s.charAt(i - 1) != '0') {
                if (i - 2 >= 0) {
                    dp[i] += dp[i - 2];
                } else {
                    dp[i]++;
                }
            }
        }
        MyUtile.dis(dp);
        return dp[n - 1];
    }

    //    //91. 解码方法 回溯做法
    public int numDecodings(String s) {
        List<List<Character>> res = new ArrayList<>();
//        System.out.println(numDecodings(s, 0, new ArrayList<>(), res, new int[s.length()], cnt));
//        System.out.println(res);
        return numDecodings(s, 0, new ArrayList<>(), res, new int[s.length()], 0);
    }

    public int numDecodings(String s, int i, List<Character> temp, List<List<Character>> res, int[] memo, Integer cnt) {
        int n = s.length(), r = 0;
        if (i == n) {
//            res.add(new ArrayList<>(temp));
            return 1;
        }
        if (memo[i] > 0) {
            return memo[i];
        }
        for (int j = i; j < n && j - i < 3; j++) {
            if (s.charAt(i) == '0') {
                continue;
            }
            int t = Integer.parseInt(s.substring(i, j + 1));
            if (t > 0 && t <= 26) {
//                temp.add((char) ('A' + t - 1));
                r += numDecodings(s, j + 1, temp, res, memo, cnt);
//                temp.remove(temp.size() - 1);
            }
        }
        return (memo[i] = r);
    }

    //
//    //131. 分割回文串 dp+回溯
    public List<List<String>> partition(String s) {
        int n = s.length();
        boolean[][] is = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    is[i][j] = true;
                    continue;
                }
                if (s.charAt(i) == s.charAt(j)) {
                    is[i][j] = j - i == 1 || is[i + 1][j - 1];
                }
            }
        }
        MyUtile.disMap(is);
        List<List<String>> res = new ArrayList<>();
        partition(s, 0, is, new ArrayList<>(), res);
        System.out.println(res);
        return res;
    }

    public void partition(String s, int i, boolean[][] is, List<String> temp, List<List<String>> res) {
        int n = s.length();
        if (i == n) {
            res.add(new ArrayList<>(temp));
        }
        for (int j = i; j < n; j++) {
            if (is[i][j]) {
                temp.add(s.substring(i, j + 1));
                partition(s, j + 1, is, temp, res);
                temp.remove(temp.size() - 1);
            }
        }
    }

    //
//    //647. 回文子串 dp
    public int countSubstrings(String s) {
        int n = s.length();
        boolean[][] res = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    res[i][j] = true;
                    continue;
                }
                if (s.charAt(i) == s.charAt(j)) {
                    String ind = (i + 1) + "," + (j - 1);
                    res[i][j] = i + 1 == j || res[i + 1][j - 1];
                }
            }
        }
        int sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < res[i].length; j++) {
                sum += res[i][j] ? 1 : 0;
            }
        }
        return sum;
    }

    //    322. 零钱兑换
    public int coinChange(int[] coins, int amount) {

        int[] dp = new int[amount + 1];
        Arrays.fill(dp, 99999);
        Arrays.sort(coins);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length && coins[j] <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        MyUtile.dis(dp);
        return dp[amount] == 99999 ? -1 : dp[amount];

    }

    //    518. 零钱兑换 II
    public int change(int amount, int[] coins) {
        int n = coins.length, i_1_j = 0;
//        int[][] dp = new int[n + 1][amount + 1];
        int[] dp1 = new int[amount + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < amount + 1; j++) {
                if (j == 0) {
//                    j = 0 代表需要凑出的目标金额为 0，那么什么都不做就是唯一的一种凑法。(坑)
                    dp1[j] = 1;
                    continue;
                }
//                dp[i][j] += dp[i - 1][j]; //dp[i - 1][j]就是上一次外循环得到的 dp[i][j]，因此这边相当于0+上一轮的dp[i][j]，而压缩成一维数组之后现在的 dp[i][j]就是上一轮留下的dp[i][j] 因此压缩后不用再加
                if (j - coins[i - 1] >= 0) {
//                    dp[i][j] += dp[i][j - coins[i - 1]];
                    dp1[j] += dp1[j - coins[i - 1]];
                }
            }
        }
//        MyUtile.dis(dp1);
        return dp1[amount];
    }

    //494. 目标和 https://blog.csdn.net/fdl123456/article/details/107054225?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169238562916800182124006%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=169238562916800182124006&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-107054225-null-null.268^v1^koosearch&utm_term=%E7%9B%AE%E6%A0%87%E5%92%8C&spm=1018.2226.3001.4450
    public int findTargetSumWays(int[] nums, int target) {
        return findTargetSumWays(nums, 0, target, new HashMap<>());
    }

    public int findTargetSumWays(int[] nums, int ind, int target, Map<Pair<Integer, Integer>, Integer> memo) {
        if (ind == nums.length) {
            return target == 0 ? 1 : 0;
        }
        if (memo.containsKey(new Pair<>(ind, target))) {
            return memo.get(new Pair<>(ind, target));
        }
        memo.put(new Pair<>(ind, target), findTargetSumWays(nums, ind + 1, target + nums[ind], memo) +
                findTargetSumWays(nums, ind + 1, target - nums[ind], memo));
        return memo.get(new Pair<>(ind, target));
    }

    //494. 目标和 （dp做法）
    public int findTargetSumWays2(int[] nums, int target) {

//        首先，如果我们把 nums 划分成两个子集 A 和 B，分别代表分配 + 的数和分配 - 的数，那么他们和 target 存在如下关系：
//        sum(A) - sum(B) = target
//        sum(A) = target + sum(B)
//        sum(A) + sum(A) = target + sum(B) + sum(A)
//        2 * sum(A) = target + sum(nums)
//        综上，可以推出 sum(A) = (target + sum(nums)) / 2，也就是把原问题转化成：nums 中存在几个子集 A，使得 A 中元素的和为 (target + sum(nums)) / 2？
        int n = nums.length;
        if (target < 0) {
//            正负都是一样的 最终就相当于是加个负号
            target = -target;
        }
        int sum = Arrays.stream(nums).sum();
//        System.out.println((-200) % 2);
        if (sum < target || (sum + target) % 2 == 1) {
            return 0;
        }
        target = (sum + target) / 2;
//        int[][] dp = new int[n + 1][target + 1];
        int[] dp1 = new int[target + 1];
//        for (int i = 0; i <= n; i++) {
//            dp[i][0] = 1;
//        }
        dp1[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = target; j >= 0; j--) {
//                dp[i][j] = dp[i - 1][j];
                if (j - nums[i - 1] >= 0) {
//                    dp[i][j] += dp[i - 1][j - nums[i - 1]];
                    dp1[j] += dp1[j - nums[i - 1]];
                }
            }
        }
//        MyUtile.disMap(dp);
        return dp1[target];

    }

    //    //279. 完全平方数 (3刷)
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, 9999);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

    //    121. 买卖股票的最佳时机 简单做法
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[] nums = new int[n];
        int min = prices[0];
        for (int i = 0; i < n; i++) {
            min = Math.min(min, prices[i]);
            nums[i] = min;
        }
        int max = 0;
        for (int i = n - 1; i >= 0; i--) {
            nums[i] = prices[i] - nums[i];
            max = Math.max(nums[i], max);
        }
        return max;

    }

    //    //    121. 买卖股票的最佳时机 状态机
////    https://labuladong.github.io/algo/di-er-zhan-a01c6/yong-dong--63ceb/yi-ge-fang-3b01b/
//    public int maxProfit(int[] prices) {
//
////        index1:第几天 index2：剩余允许的几次交易 index3：今天是否持有
//        int[][][] dp = new int[prices.length][2][2];
////        dp[-1][][0]=0;
////        dp[-1][][1]=-999999;
////        dp[][0][0]=0;
////        dp[][0][1]=-999999;
//        for (int i = 0; i < prices.length; i++) {
//            for (int j = 1; j <= 1; j++) {
////                for (int k = 0; k <= 1; k++) {
////                只需过滤掉i-1=-1的情况 因为dp[][0][0]=默认初始化的时候就是0 ，dp[][0][1]=-999999用不到
//                if (i - 1 == -1) {
//                    dp[i][j][0] = 0;
//                    dp[i][j][1] = -prices[i];
//                    continue;
//                }
////                不持有：昨天不持有今天仍然不 or 昨天持有今天卖了（卖了不意味着原来的次数是j+1 因为j是上限 不是次数 卖掉不消耗次数
////                因为这是意味着原来在不超过上限j的情况下昨天正好是持有股票的状态，然后今天卖掉了最大交易限制还是j 因为卖掉只是一次交易结束和上限没关系）两者的最大
//                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
////             持有：昨天持有今天仍然持有 or 昨天不持有今天买了（买了就说明必须消耗了一次机会 那就意味着昨天最多允许的次数肯定肯定比今天要少一次
////             因此今天消耗了一次（买入意味着百分百消耗一次 因此昨天的上限一定是少一次的 因为j代表的是上限，就是必须保证昨天少一次才能保证百分百今天买入之后的上限是j，
////             举例：如果昨天的上限是j 截止昨天之前的做了交易的次数刚好就是j次 那今天买了一次之后交易上限就是j+1次，如果截止昨天之前做了交易的次数是j-1次那刚好满足条件今天是j次））
////                因此状态是：
////                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
////                由于本题j是无限的 因此j-1=j; 所以这边的状态直接就是：
//                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j][0] - prices[i]);
////                }
//            }
//        }
//        System.out.println(dp[prices.length - 1][1][0]);
//        return dp[prices.length - 1][1][0];
//    }
//
//    //122. 买卖股票的最佳时机 II
    public int maxProfit2(int[] prices) {
//        i:day j:have?
        int n = prices.length;
        int[][] dp = new int[n + 1][2];
//        dp[0][]
        for (int i = 0; i < n + 1; i++) {
            if (i == 0) {
                dp[i][0] = 0;
                dp[i][1] = -999999;
                continue;
            }
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1]);
        }
        disMap(dp);
        return dp[n][0];
    }

    //    //    122. 买卖股票的最佳时机 II(简化版本)
    public int maxProfit22(int[] prices) {
//        i:day j:have?
        int n = prices.length;
        int[][] dp = new int[n + 1][2];
        int dpi0 = 0, dpi1 = -99999;
        for (int i = 1; i < n + 1; i++) {
            int predpi0 = dpi0;
            dpi0 = Math.max(predpi0, dpi1 + prices[i - 1]);
            dpi1 = Math.max(dpi1, predpi0 - prices[i - 1]);
        }
        disMap(dp);
        return dpi0;
    }

    //
//    //    123. 买卖股票的最佳时机 III
    public int maxProfit3(int[] prices) {
        int n = prices.length;
//        i:day j:最大交易次数上限 就是最多交易了j次，交易次数不超过j次
        int[][][] dp = new int[n + 1][3][2];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 0) {
                    dp[i][j][0] = 0;
                    dp[i][j][1] = -99999;
                    continue;
                }
                if (j == 0) {
                    dp[i][j][0] = 0;
                    dp[i][j][1] = -99999;
                    continue;
                }
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i - 1]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i - 1]);
            }
        }
        return dp[n][2][0];
    }

    //
//    //188. 买卖股票的最佳时机 IV
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        int[][][] dp = new int[n][k + 1][2];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k + 1; j++) {
                if (i == 0) {
                    dp[0][j][0] = 0;
                    dp[0][j][1] = -prices[i];
                    continue;
                }
                if (j == 0) {
                    dp[i][0][0] = 0;
                    dp[i][0][1] = -99999;
                    continue;
                }
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
            }
        }
        return dp[n - 1][k][0];

    }

    //
////309. 最佳买卖股票时机含冷冻期
//
    public int maxProfit4(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n + 1][2];
        for (int i = 0; i < n + 1; i++) {
            if (i == 0) {
                dp[i][0] = 0;
                dp[i][1] = -99999;
                continue;
            }
            if (i == 1) {
                dp[i][0] = 0;
                dp[i][1] = -prices[0];
                continue;
            }
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 2][0] - prices[i - 1]);
        }
        return dp[n][0];

    }

    //
//    //    714. 买卖股票的最佳时机含手续费
    public int maxProfit5(int[] prices, int fee) {

        int n = prices.length, k = 1;
        int[][] dp = new int[n + 1][2];
        for (int i = 0; i <= n; i++) {
            if (i == 0) {
                dp[i][0] = 0;
                dp[i][1] = -99999;
                continue;
            }
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1] - fee);
        }
        return dp[n][0];
    }

    //
//
//
//    //    剑指 Offer II 088. 爬楼梯的最少成本
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n + 1];
//        Arrays.fill(dp, 99999);
        dp[0] = dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        MyUtile.dis(dp);
        return dp[n];

    }

    //931. 下降路径最小和
    public int minFallingPathSum(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
//        int[][] res = new int[m][n];
//        res[0] = matrix[0];
        int res = 99999;
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int one = 99999, two = 99999, three = 99999;
                if (j - 1 >= 0) {
                    one = matrix[i - 1][j - 1];
                }
                two = matrix[i - 1][j];
                if (j + 1 < n) {
                    three = matrix[i - 1][j + 1];
                }
                matrix[i][j] += Math.min(Math.min(one, two), three);
            }
        }
        return Arrays.stream(matrix[m - 1]).boxed().mapToInt(Integer::intValue).min().getAsInt();
    }

    //139. 单词拆分
    public boolean wordBreak(String s, List<String> wordDict) {
        return wordBreak4(s, 0, wordDict, new HashMap<>());
    }

    public boolean wordBreak4(String s, int i, List<String> wordDict, Map<Integer, Boolean> memo) {
        int n = s.length();
        if (i == n) {
            return true;
        }
        if (memo.containsKey(i)) {
            return memo.get(i);
        }
        boolean res = false;
        for (int j = i; j <= n; j++) {
            String pre = s.substring(i, j);
            if (wordDict.contains(pre) && wordBreak4(s, j, wordDict, memo)) {
                res = true;
            }
        }
        memo.put(i, res);
        return res;
    }

    //140. 单词拆分 II
    public List<String> wordBreak2(String s, List<String> wordDict) {
        return wordBreak(s, 0, wordDict, new HashMap<Integer, List<String>>());
    }

    public List<String> wordBreak(String s, int ind, List<String> wordDict, Map<Integer, List<String>> memo) {
        List<String> res = new ArrayList<>();
        if (memo.containsKey(ind)) {
            return memo.get(ind);
        }
        if (ind == s.length()) {
            res.add("");
            return res;
        }

        for (int i = 0; ind + i <= s.length(); i++) {
            String pre = s.substring(ind, ind + i);
            if (wordDict.contains(pre)) {
                List<String> tres = wordBreak(s, ind + i, wordDict, memo);
                tres.forEach(tre -> res.add(tre.isEmpty() ? pre : pre + " " + tre));
            }
        }
        memo.put(ind, res);
        return res;
    }


    public List<String> wordBreak3(String s, List<String> wordDict) {
        return wordBreak3(s, 0, wordDict, new HashMap<>());
    }

    public List<String> wordBreak3(String s, int i, List<String> wordDict, Map<Integer, List<String>> memo) {
        List<String> res = new ArrayList<>();
        int n = s.length();
        if (i == n) {
            res.add("");
            return res;
        }
        if (memo.containsKey(i)) {
            return memo.get(i);
        }
        for (int j = i; j <= n; j++) {
            String pre = s.substring(i, j);
            if (wordDict.contains(pre)) {
                wordBreak3(s, j, wordDict, memo).forEach(e -> res.add(e.equals("") ? pre : pre + " " + e));
            }
        }
        memo.put(i, res);
        return res;
    }

    //        72. 编辑距离(hard题 多刷)
    public int minDistance(String word1, String word2) {
        return minDistance(word1, word1.length() - 1, word2, word2.length() - 1, new HashMap<>());
    }

    public int minDistance(String word1, int i, String word2, int j, Map<Pair<Integer, Integer>, Integer> memo) {
        if (i == -1) {
            return j + 1;
        }
        if (j == -1) {
            return i + 1;
        }
        if (memo.containsKey(new Pair<>(i, j))) {
            return memo.get(new Pair<>(i, j));
        }
//        相同的情况 不需要加1
        if (word1.charAt(i) == word2.charAt(j)) {
            memo.put(new Pair<>(i, j), minDistance(word1, i - 1, word2, j - 1, memo));
            return memo.get(new Pair<>(i, j));
        }
//        增删改三种取最小的方式
        memo.put(new Pair<>(i, j), Math.min(Math.min(minDistance(word1, i, word2, j - 1, memo), minDistance(word1, i - 1, word2, j - 1, memo)), minDistance(word1, i - 1, word2, j, memo)) + 1);
        return memo.get(new Pair<>(i, j));
    }

    //        72. 编辑距离（dp数组解法）
    public int minDistance2(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int i = 0; i <= n; i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
//                字符串和数组差一位所以比较的是-1的位置
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                } else {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }


    //面试题 最长回文子串
    public int longest(String s) {
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int i = 0; i < dp.length; i++) {
            Arrays.fill(dp[i], true);
        }
        for (int i = s.length() - 1; i >= 0; i--) {
            for (int j = i + 1; j < s.length(); j++) {
                dp[i][j] = Objects.equals(s.charAt(i), s.charAt(j)) && dp[i + 1][j - 1];
            }
        }
//        System.out.println(Arrays.stream(dp).collect(Collectors.toList()));
        int max = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = 0; j < s.length(); j++) {
                if (dp[i][j]) {
                    max = Math.max(max, j - i);
                }
            }
        }
        System.out.println(max);
        return max + 1;
    }

    //    //10. 正则表达式匹配
    public boolean isMatch(String s, String p) {
        return dp(s, p, 0, 0);
    }

    public boolean dp(String s, String p, int i, int j) {
        int m = s.length(), n = p.length();
        if (j == n) {
            return i == m;
        }
        if (i == m) {
            String t = p.substring(j, n);
            if (t.length() % 2 == 1) {
                return false;
            }
            for (int k = 1; k < t.length(); k += 2) {
                if (t.charAt(k) != '*') {
                    return false;
                }
            }
            return true;
        }
        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
            if (j + 1 < n && p.charAt(j + 1) == '*') {
                return dp(s, p, i + 1, j) || dp(s, p, i, j + 2);
            } else {
                return dp(s, p, i + 1, j + 1);
            }
        } else {
            if (j + 1 < n && p.charAt(j + 1) == '*') {
                return dp(s, p, i, j + 2);
            } else {
                return false;
            }
        }
    }

    //887. 鸡蛋掉落(超时版本 没优化)
    public int superEggDrop(int k, int n) {
        return dp(k, n, new HashMap<>());
    }

    //    k个鸡蛋 n个楼层（与楼层起始的层数没关系 只与楼层数量有关系，因为默认这些都是相同概率的）最少需要几次扔鸡蛋能够确定哪一层（结果F）
    public int dp(int k, int n, Map<Pair<Integer, Integer>, Integer> memo) {
        if (k == 1) {
            return n;
        }
        if (n == 0) {
            return 0;
        }
        if (memo.containsKey(new Pair<>(k, n))) {
            return memo.get(new Pair<>(k, n));
        }
        int res = 9999;
        for (int i = 1; i <= n; i++) {
//            假设在第i层扔鸡蛋，可能碎可能没碎，因此取这两种中的最大值，
//            为什么是取最大值呢？因为结果F不确定在哪一层，所以只能把两边的情况取最坏的情况，就是取次数最多的 加上这次扔鸡蛋操作+1
//            最终取最小值是因为对于所有的i都在最差的情况下如果从某一层i`开始扔下能够使得最终操作次数最少就取从i`层开始扔下
            res = Math.min(res, Math.max(dp(k - 1, i - 1, memo), dp(k, n - i, memo)) + 1);
        }
        memo.put(new Pair<>(k, n), res);
        return memo.get(new Pair<>(k, n));
    }

    //887. 鸡蛋掉落(dp数组版本)https://leetcode.cn/problems/super-egg-drop/solutions/44427/ji-ben-dong-tai-gui-hua-jie-fa-by-labuladong/
    public int superEggDrop2(int k, int n) {
//        1、无论你在哪层楼扔鸡蛋，鸡蛋只可能摔碎或者没摔碎，碎了的话就测楼下，没碎的话就测楼上。
//
//        2、无论你上楼还是下楼，总的楼层数 = 楼上的楼层数 + 楼下的楼层数 + 1（当前这层楼）。
//
//        根据这个特点，可以写出下面的状态转移方程：
//
//        dp[k][m] = dp[k][m - 1] + dp[k - 1][m - 1] + 1
//
//        dp[k][nn]=N 意思就是k个鸡蛋扔nn次最多能够测试几个楼层，只要得到最多能够测试的楼层是N层那就是结果了
//        题目的意思转变成k个鸡蛋扔nn次最多能够测试几层
//        设置n + 1长度是保证最大的下标是n，因为n是最坏的情况直接从下往上一层一层去尝试 直到最后一层次都没碎
        int[][] dp = new int[k + 1][n + 1];
        int m = 0;
        while (dp[k][m] < n) {
            m++;
            for (int i = 1; i <= k; i++) {
//                dp[i][m-1]:第i层鸡蛋没碎的话就算一下还剩i个鸡蛋操作数减一的情况下最多能够测试几层（这边得到的层数只能是i层之上的最多能够测试的层数）
//                dp[i-1][m-1]:第i层鸡蛋碎了的话就算一下还剩i-1个鸡蛋操作数减一的情况下最多能够测试几层（这边得到的层数只能是i层之下的最多能够测试的层数）
//                因为不确定是会不会碎 所以两种情况需要加起来能够得到最多的层
//                最后再加当前的这一层
                dp[i][m] = dp[i][m - 1] + dp[i - 1][m - 1] + 1;
            }
        }
        return m;
    }

    //312. 戳气球
    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[] point = new int[n + 2];
        for (int i = 0; i < n; i++) {
            point[i + 1] = nums[i];
        }
        point[0] = point[point.length - 1] = 1;

        int[][] dp = new int[n + 2][n + 2];

        for (int i = n + 1; i >= 0; i++) {
            for (int j = 0; j < n + 2; j++) {
                for (int k = i + 1; k < j; k++) {
                    dp[i][j] = Math.max(dp[i][j], dp[i][k] + dp[k][j] + point[k] * point[i] * point[j]);
                }
            }
        }
        return dp[0][n + 1];
    }

    //198. 打家劫舍
    public int rob(int[] nums) {

//        第i个房子最多得到多少
        int n = nums.length;
        if (n == 0) return 0;
        int[] dp = new int[n];
        dp[0] = nums[0];
        if (n == 1) return dp[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[n - 1];
    }

    //213. 打家劫舍 II
    public int rob2(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        if (n == 2) return Math.max(nums[0], nums[1]);
        return Math.max(robrange(nums, 0, n - 2), robrange(nums, 1, n - 1));
    }

    public int robrange(int[] nums, int l, int r) {
        int n = r - l + 1;
        int[] dp = new int[n];
        dp[0] = nums[l];
        dp[1] = Math.max(nums[l], nums[l + 1]);
        for (int i = 2; i < dp.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[l + i]);
        }
        return dp[n - 1];
    }
//337. 打家劫舍 III

    public int rob(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return root.val;
        int l = rob(root.left);
        int r = rob(root.right);
        int lc = root.left == null ? 0 : Math.max(rob(root.left.left), rob(root.left.right));
        int rc = root.right == null ? 0 : Math.max(rob(root.right.left), rob(root.right.right));
        return Math.max(root.val + lc + rc, l + r);
    }

    public static void disMap(int[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                System.out.print("(" + i + "," + j + ") " + a[i][j] + " ");
            }
            System.out.println();
        }
    }

    //    435. 无重叠区间
    public int eraseOverlapIntervals(int[][] intervals) {
        int m = intervals.length, n = intervals[0].length;
        long[][] nums = new long[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                nums[i][j] = intervals[i][j];
            }
        }
        Arrays.sort(nums, (a, b) -> {
            return (int) (a[1] - b[1]);
        });
//        disMap(nums);
        long end = nums[0][1], res = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i][0] < end) {
                res++;
            } else {
                end = nums[i][1];
            }
        }
        return (int) res;
    }

    //452. 用最少数量的箭引爆气球
    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, (a, b) -> {
            if (a[1] > b[1]) {
                return 1;
            } else if (a[1] < b[1]) {
                return -1;
            } else {
                return 0;
            }
        });
//        disMap(nums);
        int end = points[0][1], fold = 0;
        for (int i = 1; i < points.length; i++) {
            System.out.println(end);
            if (points[i][0] <= end) {
                fold++;
            } else {
                end = points[i][1];
            }
        }
        return (points.length - fold);
    }


    //
    public static void main(String[] args) {
        int[] nums = new int[]{7, 1, 5, 3, 6, 4};
        List<String> wordDict = new ArrayList<String>();
        wordDict.add("apple");
        wordDict.add("pen");
        wordDict.add("applepen");
        wordDict.add("pine");
        wordDict.add("pineapple");
        Dp dp = new Dp();
        System.out.println(dp.maxProfit2(nums));
//        System.out.println(dp.numDecodings2("2611055971756562"));
//        System.out.println(dp.partition("aab"));
//        System.out.println(dp.findTargetSumWays2(nums, 3));
//        System.out.println(dp.change(5, nums));
//        System.out.println(dp.canPartition(nums));
//        System.out.println(dp.longestPalindromeSubseq("bbbab"));
//        System.out.println(dp.longestPalindrome("cbbd"));
//        System.out.println(dp.wordBreak3("pineapplepenapple", wordDict));
//        System.out.println(dp.coinChange(nums, 11));
//        System.out.println(dp.climbStairs(3));

//        dp.lengthOfLIS(nums);

//        dp.maxSubArray(new int[]{5, 4, -1, 7, 8});
//        System.out.println(dp.minDistance4("leetcode", "etco"));
//        dp.jump(new int[]{2, 3, 0, 1, 4});
//        char[][] c = new char[2][2];
//        c = new char[][]{{'0', '1'}, {'1', '0'}};
//        dp.maximalSquare(c);

//        dp.canPartition(new int[]{
//                1, 5, 10, 6
//        });
//        dp.partition("aab");
//        dp.numSquares(12);
//        dp.isMatch("aa", "a*");
    }
}
