package category;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class Dp {
    public int climbStairs(int n) {
        if (n < 3) {
//            边界条件 刚好为n；
            return n;
        }


        int[] stairs = new int[n + 1];
        stairs[1] = 1;
        stairs[2] = 2;
        for (int i = 3; i <= n; i++) {
            stairs[i] = stairs[i - 1] + stairs[i - 2];
        }
        return stairs[n];

    }

    //    300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length + 1];
        int max = 1;
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (nums[i] > nums[j]) {
                    max = Math.max(max, dp[i] = Math.max(dp[j] + 1, dp[i]));
                }
            }
        }
        return max;
    }

    //53. 最大子数组和
    public int maxSubArray(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
//        int max=-99999;
//        以num[i]结尾的最大连续子数组
        int[] dp = new int[nums.length];
        Arrays.fill(dp, -99999);
        dp[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
        }
        new SortTest().dis(dp);
//        System.out.println(dp);

        return Arrays.stream(dp).max().getAsInt();
    }

    //1143. 最长公共子序列 备忘录+dp递归
//    https://labuladong.github.io/algo/di-er-zhan-a01c6/zi-xu-lie--6bc09/dong-tai-g-c481e/
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] m = new int[text1.length()][text2.length()];
        for (int i = 0; i < m.length; i++) {
            Arrays.fill(m[i], -1);
        }
        return dp(text1, 0, text2, 0, m);
    }

    //    dp方法的含义就是text1[i~text1.length()-1]和text2[j~text2.length()-1]的最长公共子序列
    int dp(String text1, int i, String text2, int j, int[][] m) {
        if (i >= text1.length() || j >= text2.length()) return 0;
        if (m[i][j] != -1) return m[i][j];
//        只有当相等时lcs加一 其他情况都是交给后面没有比对的序列去比对
        if (text1.charAt(i) == text2.charAt(j)) {
            m[i][j] = 1 + dp(text1, i + 1, text2, j + 1, m);
        } else {
            m[i][j] = Math.max(dp(text1, i + 1, text2, j, m), dp(text1, i, text2, j + 1, m));
        }
        return m[i][j];
    }


    //583. 两个字符串的删除操作 求出最长公共字串之后各减去这个lcs长度
    public int minDistance(String word1, String word2) {
        int[][] m = new int[word1.length()][word2.length()];
        for (int i = 0; i < m.length; i++) {
            Arrays.fill(m[i], -1);
        }
        int lcslen = dp2(word1, 0, word2, 0, m);
        return word1.length() + word2.length() - 2 * lcslen;
    }

    public int dp2(String word1, int i, String word2, int j, int[][] m) {
        if (i >= word1.length() || j >= word2.length()) {
            return 0;
        }
        if (m[i][j] != -1) {
            return m[i][j];
        }
        return m[i][j] = word1.charAt(i) == word2.charAt(j) ? 1 + dp2(word1, i + 1, word2, j + 1, m) : Math.max(dp2(word1, i + 1, word2, j, m), dp2(word1, i, word2, j + 1, m));
    }

    //55. 跳跃游戏
    public boolean canJump(int[] nums) {

//        dp[i]:到当前位置最小次数
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 99999);
        dp[0] = 1;
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] >= i - j) {
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
        new SortTest().dis(dp);
        return dp[nums.length - 1] != 99999;

    }

    //45. 跳跃游戏 II
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
        new SortTest().dis(dp);
        return dp[nums.length - 1];

    }
//62. 不同路径

    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 1;
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        for (int i = 0; i < dp.length; i++) {
            new SortTest().dis(dp[i]);
        }
        return dp[m - 1][n - 1];

    }

    //64. 最小路径和
    public int minPathSum(int[][] grid) {

        int m = grid.length, n = grid[0].length;
////        if
//        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) continue;
                if (i == 0) {
                    grid[i][j] += grid[i][j - 1];
                    continue;
                }
                if (j == 0) {
                    grid[i][j] += grid[i - 1][j];
                    continue;
                }
//                if (i == 0 || j == 0) continue;
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(grid[i][j] + " ");
            }
            System.out.println();
        }

        return grid[m - 1][n - 1];
    }

    //152. 乘积最大子数组
    public int maxProduct(int[] nums) {

//        dp[i]:当前乘积算进去的最大
//        int[] dp = new int[nums.length];
//        Arrays.fill(dp, 1);
        int[] min = new int[nums.length];
        int[] max = new int[nums.length];
        Arrays.fill(min, 99999);
        Arrays.fill(max, -99999);
        min[0] = nums[0];
        max[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {

//            dp[i]= Math.max(dp[i-1]*nums[i],nums[i]);
            int t = Math.min(Math.min(min[i - 1] * nums[i], nums[i]), max[i - 1] * nums[i]);
            max[i] = Math.max(Math.max(max[i - 1] * nums[i], nums[i]), min[i - 1] * nums[i]);
            min[i] = t;
        }
//        new SortTest().dis(min);
//        new SortTest().dis(max);
        return Arrays.stream(max).max().getAsInt();
    }

    //221. 最大正方形
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
//        以dp[i][j]:以ij这个点作为正方形的右下角的正方形的最大边数 前提是dp[i][j]必须是1且作为正方形右下角
//        意味着ij左边左上上面只要有一个==0就直接是1
        int[][] dp = new int[m][n];
        int max = -999999;
//        dp[0][0]=matrix[0][0];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = matrix[i][j] - '0';
                    max = Math.max(max, dp[i][j]);
                    continue;
                }

                if (matrix[i][j] == '1') {
                    if (dp[i - 1][j - 1] != 0 && dp[i][j - 1] != 0 && dp[i - 1][j] != 0) {
//                        取最短是因为往左边左上上面这三个地方扩散能够扩散多大取决于最短的那个 因为是正方形 所以只能取最短的 如果是长方形那就不一定了
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
                    } else {
                        dp[i][j] = 1;
                    }
                } else {
                    dp[i][j] = 0;
                }
                max = Math.max(max, dp[i][j]);
            }
        }
        for (int i = 0; i < m; i++) {
            new SortTest().dis(dp[i]);
        }
        System.out.println(max);
        return max * max;
    }

    //416. 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        if (sum % 2 != 0) {
            return false;
        }
        int target = sum / 2;
        int[][] dp = new int[target + 1][nums.length];
//        dp[0] = 0;
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (j == 0) {
                    dp[i][j] = i >= nums[j] ? nums[j] : 0;
                    continue;
                }
                if (i - nums[j] >= 0) {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - nums[j]][j - 1] + nums[j]);
                } else {
                    dp[i][j] = dp[i][j - 1];
                }

                if (dp[i][j] == target) return true;
            }
//            new SortTest().dis(dp[i]);
        }
//        for (int i = 0; i <= target; i++) {
//            new SortTest().dis(dp[i]);
//        }

        return false;

    }

    //91. 解码方法
    public int numDecodings(String s) {
        int n = s.length();
        //        0~n-1字符的解码方法数 注意是n-1 因为假如dp[1]的含义是前面1个字符 就是s[0] dp[2]的含义是前面2个字符 就是s[0~1] 因此这是不得不这么做的
        int[] dp = new int[n + 1];
//        前0个代表空串 空串也算一种
        dp[0] = 1;

        for (int i = 1; i <= n; i++) {
//            对i - 1做判断实际就是前i个字符做判断 i-2同理
            if (!Objects.equals(s.charAt(i - 1), '0')) {
                dp[i] += dp[i - 1];
            }
            if (i > 1 && !Objects.equals(s.charAt(i - 2), '0') && 10 * (s.charAt(i - 2) - '0') + s.charAt(i - 1) - '0' <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }

    //131. 分割回文串 dp+回溯
    public List<List<String>> partition(String s) {
//        dp回文 is[i][j] mean s[i~j]是不是回文串
        boolean[][] is = new boolean[s.length()][s.length()];
//        循环体注意 is[i + 1][j - 1] 是is[i][j] 的左下角 不管怎么要保证左下角提前产生
        for (int i = s.length() - 1; i >= 0; i--) {// 从下往上保证下面的i + 1可以提前产生
            for (int j = 0; j < s.length(); j++) {// 从小到大保证下面的j - 1可以提前产生
                if (i >= j) {
                    is[i][j] = true;
                } else {
                    is[i][j] = is[i + 1][j - 1] && Objects.equals(s.charAt(i), s.charAt(j));
                }
            }
        }
        MyUtile.disMap(is);

        List<List<String>> res = new ArrayList<>();
        dfs(s, 0, new ArrayList<>(), res, is, 0);
        for (int i = 0; i < res.size(); i++) {
            System.out.println(res.get(i));
        }
        return res;
    }

    //   dfs求s[begin~n]的回文字符串列表
//    int level=0;
    public void dfs(String s, int begin, List<String> tmp, List<List<String>> res, boolean[][] is, int level) {
//        MyUtile.printBlank(level++,"begin="+begin);
        if (begin == s.length()) {
            res.add(new ArrayList<>(tmp));
//            for (int i = 0; i < res.size(); i++) {
//                System.out.println(res.get(i));
//            }

//            System.out.println(tmp);
//            MyUtile.printReturn(level-1,null);
            return;
        }
        for (int i = begin; i < s.length(); i++) {
            if (is[begin][i]) {
                tmp.add(s.substring(begin, i + 1));
                dfs(s, i + 1, tmp, res, is, level);
                tmp.remove(tmp.size() - 1);
            }
        }
//        MyUtile.printBlank(--level,"begin="+begin);
    }

    //647. 回文子串 dp
    public int countSubstrings(String s) {
        int count = 0;
        int[][] is = new int[s.length()][s.length()];
        for (int i = 0; i < is.length; i++) {
            Arrays.fill(is[i], 1);
        }
        for (int i = s.length() - 1; i >= 0; i--) {
            for (int j = i + 1; j < s.length(); j++) {
                is[i][j] = Objects.equals(s.charAt(i), s.charAt(j)) && is[i + 1][j - 1] == 1 ? 1 : 0;
            }
        }
        for (int i = 0; i < is.length; i++) {
            for (int j = i; j < is[0].length; j++) {
                count += is[i][j];
            }
        }
        System.out.println(count);
//        MyUtile.disMap(is);
        return count;
    }

    //279. 完全平方数
    public int numSquares(int n) {
//        n的最少平方数的个数
        int[] dp = new int[n + 1];
//        Arrays.fill(dp, 1);
        dp[0] = 0;

        for (int i = 1; i <= n; i++) {
            int min = 99999;
            for (int j = 1; j * j <= i; j++) {
                min = Math.min(min, dp[i - j * j]);
            }
//            +1的意思是上面的min是选中了某一个j是满足条件的 无论如何尽管别的j没有满足条件至少有一个j=1是满足条件的 因此dp记数要加一
            dp[i] = min + 1;
        }
        System.out.println(dp[n]);
        return dp[n];
    }


    //    121. 买卖股票的最佳时机 状态机
//    https://labuladong.github.io/algo/di-er-zhan-a01c6/yong-dong--63ceb/yi-ge-fang-3b01b/
    public int maxProfit(int[] prices) {

//        index1:第几天 index2：剩余允许的几次交易 index3：今天是否持有
        int[][][] dp = new int[prices.length][2][2];
//        dp[-1][][0]=0;
//        dp[-1][][1]=-999999;
//        dp[][0][0]=0;
//        dp[][0][1]=-999999;
        for (int i = 0; i < prices.length; i++) {
            for (int j = 1; j <= 1; j++) {
//                for (int k = 0; k <= 1; k++) {
//                只需过滤掉i-1=-1的情况 因为dp[][0][0]=默认初始化的时候就是0 ，dp[][0][1]=-999999用不到
                if (i - 1 == -1) {
                    dp[i][j][0] = 0;
                    dp[i][j][1] = -prices[i];
                    continue;
                }
//                不持有：昨天不持有今天仍然不 or 昨天持有今天卖了 两者的最大
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
//             不持有：昨天持有今天仍然持有 or 昨天不持有今天买了（买了就说明消耗了一次机会 那就意味着昨天最多允许的次数肯定肯定比今天要少一次
//             因此今天消耗了一次 因为昨天允许的次数不一定要全用完 我看比今天少一次的昨天dp 这样能够保证取到的是昨天的最大值
//             例如昨天是dp[i - 1][3][0]意思是昨天最多允许的交易次数为3次 但是不意味着截止昨天一定是做了3次交易才是最大值 可能做了一次才是最大值
//             但对今天来说我是取最大限制为3次的最大值 就是dp[i - 1][3][0]的值） 两者的最大
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
//                }
            }
        }
        System.out.println(dp[prices.length - 1][1][0]);
        return dp[prices.length - 1][1][0];
    }

    //122. 买卖股票的最佳时机 II
    public int maxProfit2(int[] prices) {
        int[][] dp = new int[prices.length][2];
        for (int i = 0; i < prices.length; i++) {
            if (i - 1 == -1) {
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[prices.length - 1][0];

    }
//    剑指 Offer 10- II. 青蛙跳台阶问题

//    public int numWays(int n) {
//
//    }

    //    剑指 Offer II 088. 爬楼梯的最少成本
    public int minCostClimbingStairs(int[] cost) {
        int[] dp = new int[cost.length];
        dp[0] = cost[0];
        dp[1] = cost[1];
        for (int i = 2; i < cost.length; i++) {
            dp[i] = Math.min(dp[i - 1], dp[i - 2]) + cost[i];
        }
        System.out.println(Arrays.stream(dp).boxed().collect(Collectors.toList()));
        return Math.min(dp[cost.length - 1], dp[cost.length - 2]);

    }

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

    //10. 正则表达式匹配
    public boolean isMatch(String s, String p) {
        return dp(s, p, 0, 0);
    }

    public boolean dp(String s, String p, int i, int j) {
        int m = s.length(), n = p.length();
        if (j == n) {
            return i == m;
        }
        if (i == m) {
            if ((n - j) % 2 == 1) {
                return false;
            }
            for (int k = j; k < n; k += 2) {
                if (p.charAt(k + 1) != '*') {
                    return false;
                }
            }
            return true;
        }

        if (Objects.equals(s.charAt(i), p.charAt(j)) || Objects.equals(p.charAt(j), '.')) {
            if (j + 1 < n && Objects.equals(p.charAt(j + 1), '*')) {
                return dp(s, p, i, j + 2) || dp(s, p, i + 1, j);
            } else {
                return dp(s, p, i + 1, j + 1);
            }
        } else {
            if (j + 1 < n && Objects.equals(p.charAt(j + 1), '*')) {
                return dp(s, p, i, j + 2);
            } else {
                return false;
            }
        }
    }

    public static void main(String[] args) {
        Dp dp = new Dp();
//        System.out.println(dp.climbStairs(3));

//        dp.lengthOfLIS(new int[]{7,7,7,7,7,7,7});

//        dp.maxSubArray(new int[]{5, 4, -1, 7, 8});
//        System.out.println(dp.minDistance("abcde", "def"));
//        dp.jump(new int[]{2, 3, 0, 1, 4});
//        char[][] c = new char[2][2];
//        c = new char[][]{{'0', '1'}, {'1', '0'}};
//        dp.maximalSquare(c);

//        dp.canPartition(new int[]{
//                1, 5, 10, 6
//        });
//        dp.partition("aab");
//        dp.numSquares(12);
        dp.isMatch("aa","a*");
    }
}
