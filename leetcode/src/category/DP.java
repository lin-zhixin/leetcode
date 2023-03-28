package category;

import java.util.*;

public class DP {
    //5. 最长回文子串 动态规划
//    https://leetcode.cn/problems/longest-palindromic-substring/solution/zhong-xin-kuo-san-fa-he-dong-tai-gui-hua-by-reedfa/
    public String longestPalindrome(String s) {
        System.out.println(s);
        int start = 0, end = 0, maxlen = 1, strlen = s.length();
        if (s.length() < 2) return s.substring(start, end + 1);

        boolean[][] dp = new boolean[strlen][strlen];
        for (int r = 1; r < s.length(); r++) {
            for (int l = 0; l < r; l++) {
                if (s.charAt(l) == s.charAt(r) && (r - l <= 2 || dp[l + 1][r - 1])) {
                    dp[l][r] = true;
                    if (r - l + 1 > maxlen) {
                        maxlen = r - l + 1;
                        start = l;
                        end = r;
                    }
                }
            }
        }
//        for (int i = 0; i < strlen; i++) {
////                System.out.println(Arrays.stream(dp[i]));
//            for (int j = 0; j < strlen; j++) {
//                System.out.print((dp[i][j] ? 1 : 0) + " ");
//            }
//            System.out.println();
//        }
        return s.substring(start, end + 1);
    }

    //    45. 跳跃游戏 II
//    public int jump(int[] nums) {
//
////        dp[i][j]=min(dp[i][j-1]+1,dp[i][j])
////        初始化dp[i][j]为下标非0的话就是下标否则是
//
//    }
//322. 零钱兑换
    public int coinChange(int[] coins, int amount) {
//       下标：金额； 值：所属要的最少的硬币数
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE - 100);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
//                System.out.println(i+"->"+dp[i]);
            }
            IntUtil.dis(dp);
        }
        return dp[amount] == Integer.MAX_VALUE - 100 ? -1 : dp[amount];
    }

    //    139. 单词拆分
    public boolean wordBreak(String s, List<String> wordDict) {

        Set<String> words = new HashSet<>(wordDict);

        boolean[] dp = new boolean[s.length() + 1];//dp=1说明s[0~(i-1)就是前面长度为i的串能够拼出来
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    //    139. 单词拆分(解法2 递归回溯 动态规划)
    public boolean wordBreak2(String s, List<String> wordDict) {
        // 备忘录，-1 代表未计算，0 代表无法凑出，1 代表可以凑出
        int[] memo = new int[s.length()];
        Arrays.fill(memo, -1);
        System.out.println(dp(s, 0, wordDict, memo));
        return dp(s, 0, wordDict, memo);

    }

    // s[i~]这个串能否被拼出 注意是i之后的子串
    public boolean dp(String s, int i, List<String> wordDict, int[] memo) {
//        空字符串 能被拼出
        if (i == s.length()) {
            return true;
        }
        if (memo[i] != -1) {
            return memo[i] != 0;
        }
        // 遍历当前序列
        int len = 1;
        while (len + i <= s.length()) {
            // 找到s[i~]的前缀：s[i~len]，如果s[i~i+len]存在那问题就变成了s[i+len~]是否存在的子问题
            if (wordDict.contains(s.substring(i, i + len)) && dp(s, i + len, wordDict, memo)) {
                memo[i] = 1;
                return true;
            }
            len++;
        }
//        都遍历完了还是没有说明不存在 就是i之后的字符串拼不成0
        memo[i] = 0;
        return false;
    }

    public List<String> wordBreak3(String s, List<String> wordDict) {
        List<String> res = new ArrayList<>();
        // Map用来防止重复计算 Integer:下标索引 List<String>：s[Integer~]子串所组成的所有结果
        Map<Integer, List<String>> memo = new HashMap<>();

        System.out.println(dp2(s, 0, wordDict, memo));
        return dp2(s, 0, wordDict, memo);
    }

    // s[i~]这个串被拼出的所有可能值
    public List<String> dp2(String s, int i, List<String> wordDict, Map<Integer, List<String>> memo) {
        List<String> res = new ArrayList<>();
//        空字符串 能被拼出
        if (i == s.length()) {
//            必要的 空字符串必须返回一个 "" 否则被认为s[s.length()~]不能被拼出来，之后res永远为空
            res.add("");//保证res有长度 如果什么都不做下面的for (String right : tempRes)没有值
            return res;
        }
        if (memo.containsKey(i)) {
            return memo.get(i);
        }
        // 遍历当前序列
        int len = 1;
        while (len + i <= s.length()) {
            // 找到s[i~]的前缀：s[i~len]，如果s[i~i+len]存在那问题就变成了s[i+len~]是否存在的子问题
            String left = s.substring(i, i + len);
            if (wordDict.contains(left)) {
//                rights :s[i~i+len]的所有可能组合，只要加上当前的前缀就是s[i~]的所有可能组合
                List<String> tempRes = dp2(s, i + len, wordDict, memo);
                for (String right : tempRes) {
                    res.add(right.isEmpty() ? left : left + " " + right);
                }
                memo.put(i, res);
            }
            len++;
        }
        return res;
    }

    //300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
//        dp[i]:前index个序列的最长子序列长度
        int[] dp = new int[nums.length + 1];
        int max = 1;
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                    max = Math.max(max, dp[i]);
                }
            }
        }
        return max;
    }

    public static void main(String[] args) {
        DP o = new DP();
        System.out.println(o.lengthOfLIS(new int[]{0, 1, 0, 3, 2, 3}));
//        List<String> ss = new ArrayList<>();
//        ss.add("cat");
//        ss.add("cats");
//        ss.add("and");
//        ss.add("sand");
//        ss.add("dog");
//        o.wordBreak3("catsanddog", ss);

    }
}
