package test2;

import category.MyUtile;
import javafx.util.Pair;

import java.util.*;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class HardTest2 {


    //    25. K 个一组翻转链表
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode a = head, b = head;
        int t = k;
        while ((t = t - 1) >= 0 && b != null) {
            b = b.next;
        }
        if (b == null && t >= 0) {
            return head;
        }
        ListNode newHead = reverse(a, b);
        a.next = reverseKGroup(a.next, k);
        return newHead;

    }

    public ListNode reverse(ListNode a, ListNode b) {
//        [)
        ListNode pre = b, p = a;
        while (p != b) {
            ListNode next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        return pre;
    }

    // 非递归
    public ListNode reverseKGroup2(ListNode head, int k) {
        ListNode pre = new ListNode(), a = head, b = head, newHead = pre;
        pre.next = a;
        while (a != null) {
            int t = k;
            while ((t = t - 1) >= 0 && b != null) {
                b = b.next;
            }
            if (t >= 0) {
                break;
            }
            pre.next = reverse(a, b);
            pre = a;
            a = b;
        }
        return newHead.next;

    }

    //23. 合并 K 个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        return mergeKLists(lists, 0, lists.length - 1);

    }

    public ListNode mergeKLists(ListNode[] lists, int l, int r) {
        if (l < r) {
            int mid = l + (r - l) / 2;
            return merge(mergeKLists(lists, l, mid), mergeKLists(lists, mid + 1, r));
        }
        return l == r ? lists[l] : null;
    }

    public ListNode merge(ListNode a, ListNode b) {
        ListNode newhead = new ListNode(), res = newhead;
//        newhead.next=
        while (a != null && b != null) {
            if (a.val <= b.val) {
                newhead.next = a;
                a = a.next;
            } else {
                newhead.next = b;
                b = b.next;
            }
            newhead = newhead.next;
        }
        if (a != null) {
            newhead.next = a;
        }
        if (b != null) {
            newhead.next = b;

        }
        return res.next;
    }

    //42. 接雨水
    public int trap(int[] height) {
        int l = 0, r = height.length - 1, lmax = -9999, rmax = -9999, sum = 0;
        while (l < r) {
            lmax = Math.max(lmax, height[l]);
            rmax = Math.max(rmax, height[r]);
            sum += height[l] <= height[r] ? Math.min(lmax, rmax) - height[l++] : Math.min(lmax, rmax) - height[r--];
        }
        return sum;
    }

    //124. 二叉树中的最大路径和
    int maxPathSumx = -99999;

    public int maxPathSum(TreeNode root) {
        maxPathSum2(root);
        return maxPathSumx;
    }

    public int maxPathSum2(TreeNode root) {
        if (root == null) {
            return -99999;
        }
        int lm = maxPathSum2(root.left);
        int rm = maxPathSum2(root.right);
        PriorityQueue<Integer> heap = new PriorityQueue<>(Comparator.reverseOrder());
        heap.add(root.val);
        heap.add(root.val + lm);
        heap.add(root.val + rm);
        heap.add(root.val + rm + lm);
        int res = heap.peek();
        heap.add(lm);
        heap.add(rm);
        maxPathSumx = Math.max(maxPathSumx, heap.peek());
        return res;

    }

    //4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        if ((m + n) % 2 == 0) {
            return (findMedianSortedArraysKTH(nums1, nums2, (m + n) / 2) + findMedianSortedArraysKTH(nums1, nums2, (m + n) / 2 + 1)) / 2.0;
        } else {
            return findMedianSortedArraysKTH(nums1, nums2, (m + n) / 2 + 1);
        }
    }

    public double findMedianSortedArraysKTH(int[] nums1, int[] nums2, int k) {

        int i = 0, j = 0, m = nums1.length, n = nums2.length;
        while (true) {
            if (i == m) {
                return nums2[j + k - 1];
            }
            if (j == n) {
                return nums1[i + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[i], nums2[j]);
            }

            int half = k / 2, ti = Math.min(i + half, m) - 1, tj = Math.min(j + half, n) - 1;
            if (nums1[ti] <= nums2[tj]) {
                k -= (ti - i + 1);
                i = ti + 1;
            } else {
                k -= (tj - j + 1);
                j = tj + 1;
            }
        }
    }

    //239. 滑动窗口最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        Deque<Integer> q = new ArrayDeque<>();
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            while (!q.isEmpty() && q.peekLast() < nums[i]) {
                q.pollLast();
            }
            q.offerLast(nums[i]);


            if (i >= k) {
                if (!q.isEmpty() && nums[i - k] == q.peekFirst()) {
                    q.pollFirst();
                }
            }
            if (i >= k - 1) {
                res.add(q.peekFirst());

            }
        }
        return res.stream().mapToInt(Integer::intValue).toArray();

    }

    //76. 最小覆盖子串
    public String minWindow(String s, String t) {
        Map<Character, Integer> win = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            need.put(t.charAt(i), need.getOrDefault(t.charAt(i), 0) + 1);
        }
        int l = 0, r = 0, valid = 0, start = 0, len = 99999;
        while (r < s.length()) {
            Character c = s.charAt(r++);
            if (need.containsKey(c)) {
                win.put(c, win.getOrDefault(c, 0) + 1);
                if (Objects.equals(win.get(c), need.get(c))) {
                    valid++;
                }
                while (valid == need.size()) {
                    if (r - l < len) {
                        start = l;
                        len = r - l;
                    }

                    Character lc = s.charAt(l++);
                    if (win.containsKey(lc)) {
                        if (Objects.equals(win.get(lc), need.get(lc))) {
                            valid--;
                        }
                        win.put(lc, win.get(lc) - 1);
                    }

                }

            }
        }
        return len == 99999 ? "" : s.substring(start, start + len);
    }

    //32. 最长有效括号
    public int longestValidParentheses(String s) {
        int[] dp = new int[s.length()];
        int res = 0;
        Stack<Integer> stk = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stk.push(i);
            } else {
                if (!stk.isEmpty()) {
                    int l = stk.pop();
                    dp[i] = i - l + 1 + (l == 0 ? 0 : dp[l - 1]);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    public int longestValidParentheses2(String s) {
        int res = 0, l = 0, r = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                l++;
            } else {
                r++;
            }
            if (r == l) {
                res = Math.max(res, r * 2);
            } else if (r > l) {
                l = 0;
                r = 0;
            }
        }
        l = 0;
        r = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            char c = s.charAt(i);
            if (c == '(') {
                l++;
            } else {
                r++;
            }
            if (r == l) {
                res = Math.max(res, l * 2);
            } else if (l > r) {
                l = 0;
                r = 0;
            }
        }
        return res;
    }

    //224. 基本计算器
    public int calculate(String s) {
        Deque<Character> q = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) != ' ') {
                q.offerLast(s.charAt(i));
            }
        }
        return calculate2(q);

    }

    public int calculate2(Deque<Character> q) {
        Stack<Integer> stack = new Stack<>();
        Integer num = 0;
        char sig = '+';
        while (!q.isEmpty()) {
            char c = q.poll();
            if (Character.isDigit(c)) {
                num = num * 10 + (c - '0');
            }
            if (!Character.isDigit(c) || q.isEmpty()) {
                if (c == '(') {
                    num = calculate2(q);
                }
                switch (sig) {
                    case '+': {
                        stack.push(num);
                        break;
                    }
                    case '-': {
                        stack.push(-num);
                        break;
                    }
                    case '*': {
                        stack.push(stack.pop() * num);
                        break;
                    }
                    case '/': {
                        stack.push(stack.pop() / num);
                    }
                }
                if (c == ')') {
                    break;
                }
                sig = c;
                num = 0;
            }
        }
        System.out.println(stack);
        return stack.stream().mapToInt(Integer::intValue).sum();
    }

    //    LCR 170. 交易逆序对的总数
    int[] cnt;

    public int reversePairs(int[] record) {
        int n = record.length;
        cnt = new int[n];
        Pair<Integer, Integer>[] nums = new Pair[n];
        for (int i = 0; i < record.length; i++) {
            nums[i] = new Pair<>(record[i], i);
        }
        mergeS(nums, 0, n - 1);
        Arrays.stream(cnt).boxed().collect(Collectors.toList());
        return Arrays.stream(cnt).sum();
    }

    public void mergeS(Pair<Integer, Integer>[] nums, int l, int r) {
        if (l < r) {
            int mid = l + (r - l) / 2;
            mergeS(nums, l, mid);
            mergeS(nums, mid + 1, r);
            merge(nums, l, mid, r);
        }
    }

    public void merge(Pair<Integer, Integer>[] nums, int l, int mid, int r) {
        Deque<Pair<Integer, Integer>> q = new LinkedList<>();
        int p1 = l, p2 = mid + 1;
        while (p1 <= mid && p2 <= r) {
            if (nums[p1].getKey() <= nums[p2].getKey()) {

                cnt[nums[p1].getValue()] += (p2 - (mid + 1));
                q.offerLast(nums[p1++]);
            } else {
                q.offerLast(nums[p2++]);
            }
        }
        while (p1 <= mid) {
            cnt[nums[p1].getValue()] += (p2 - (mid + 1));
            q.offerLast(nums[p1++]);
        }
        while (p2 <= r) {
            q.offerLast(nums[p2++]);
        }
        for (int i = l; i <= r; i++) {
            nums[i] = q.poll();
        }

    }

    //135. 分发糖果
    public int candy(int[] ratings) {
        int n = ratings.length, incLen = 1, decLen = 0, res = 1;
        for (int i = 1; i < n; i++) {
//            大于或者等于都视为递增 等于相当于重新开始一个递增序列
            if (ratings[i] >= ratings[i - 1]) {
//                如果decLen>0说明i - 1是递减序列的最后一个，decLen还没被置0，也说明i是新的递增序列的第一个
                if (decLen > 0) {
                    incLen = 1;
                }
                decLen = 0;
//                ratings[i] == ratings[i - 1] 相当于从i处重新开始一个递增序列，也说明i是新的递增序列的第一个
                incLen = ratings[i] == ratings[i - 1] ? 1 : incLen + 1;
                res += incLen;
            } else {
                decLen++;
                if (incLen == decLen) {
                    decLen++;
                }
                res += decLen;
            }
        }
        return res;

    }

    //10. 正则表达式匹配
    public boolean isMatch(String s, String p) {
        return isMatch(s, p, 0, 0);
    }

    public boolean isMatch(String s, String p, int i, int j) {
        int m = s.length(), n = p.length();
        if (j == n) {
            return i == m;
        }
        if (i == m) {
            if ((n - j) % 2 == 1) {
                return false;
            }
            for (int k = j + 1; k < n; k += 2) {
                if (p.charAt(k) != '*') {
                    return false;
                }
            }
            return true;
        }
        Character sc = s.charAt(i), pc = p.charAt(j);
        if (sc == pc || pc == '.') {
            if (j < n - 1 && p.charAt(j + 1) == '*') {
                return isMatch(s, p, i + 1, j) || isMatch(s, p, i, j + 2);
            } else {
                return isMatch(s, p, i + 1, j + 1);
            }
        } else {
            if (j < n - 1 && p.charAt(j + 1) == '*') {
                return isMatch(s, p, i, j + 2);
            } else {
                return false;
            }
        }

    }

    //44. 通配符匹配
    public boolean isMatch2(String s, String p) {
        int m = s.length(), n = p.length();
        int[][] memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            Arrays.fill(memo[i], -1);
        }
        return isMatch2(s, p, 0, 0, memo) == 1;
    }

    public int isMatch2(String s, String p, int i, int j, int[][] memo) {
        int m = s.length(), n = p.length();
        if (j == n) {
            System.out.println(i);
            return i == m ? 1 : 0;
        }
        if (i == m) {
            System.out.println(j);
            for (int k = j; k < n; k++) {
                if (p.charAt(k) != '*') {
                    return 0;
                }
            }
            return 1;
        }
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        char sc = s.charAt(i), pc = p.charAt(j);
        if (pc == '?' || sc == pc) {
            return memo[i][j] = isMatch2(s, p, i + 1, j + 1, memo);
        } else {
            if (pc == '*') {
                return memo[i][j] = (isMatch2(s, p, i + 1, j, memo) == 1 || isMatch2(s, p, i, j + 1, memo) == 1) ? 1 : 0;
            } else {
                return 0;
            }
        }

    }


    //    887. 鸡蛋掉落
    public int superEggDrop(int k, int n) {
//        dp[k][n]:k个鸡蛋，允许操作n次最多能够检测几层
        int[][] dp = new int[k + 1][n + 1];
        int j;
//        dp[k][j - 1] 应该用j-1比较，因为j-1才是上一轮得到的结果
        for (j = 1; dp[k][j - 1] < n; j++) {
            System.out.println(dp[k][j]);
            for (int i = 1; i <= k; i++) {
//                每次操作，要么碎要么不碎，睡了就是只需要检测当前层下面的，不碎就是上面的，所以把下面的最大层数+上面最大层数相加再加当前这层
//                如果结果大于n也就是最少需要操作的次数，
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j - 1] + 1;
            }
        }
//        MyUtile.disMap(dp);
        return j - 1;

    }

    //312. 戳气球
    public int maxCoins(int[] nums) {
        int n = nums.length;
        List<Integer> list = Arrays.stream(nums).boxed().collect(Collectors.toList());
        list.add(0, 1);
        list.add(1);
        System.out.println(list);
        nums = list.stream().mapToInt(Integer::intValue).toArray();
        MyUtile.dis(nums);
//        dp[i][j]:(i,j)闭区间的最大值，也就是i+1~j-1的最大值
        int[][] dp = new int[n + 2][n + 2];
        for (int i = n + 1; i >= 0; i--) {
            for (int j = i + 1; j < n + 2; j++) {
                for (int k = i + 1; k < j; k++) {
//                    因为i<k<j 所以dp[i][k]是dp[i][j]的下边几列的元素，同理dp[k][j]是dp[i][j]的左边几列行元素
//                    所以需要先遍历大于i的和小于j的 所以i从n+1到0，j从(i+1)到n+1 注意：这边是开区间，sy所以k没必要从i开始
                    dp[i][j] = Math.max(dp[i][j], dp[i][k] + dp[k][j] + (nums[i] * nums[k] * nums[j]));
                }
            }
        }
        MyUtile.disMap(dp);
        return dp[0][n + 1];

    }

    //84. 柱状图中最大的矩形
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n], right = new int[n];

        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) {
                stack.pop();
            }
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        stack.clear();
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) {
                stack.pop();
            }
            right[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
//        MyUtile.dis(left);
//        MyUtile.dis(right);
//        System.out.println(stack);
        int res = 0;
        for (int i = 0; i < n; i++) {
            res = Math.max(res, (right[i] - left[i] - 1) * heights[i]);
        }
        System.out.println(res);
        return res;
    }

    //85. 最大矩形
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int m = matrix.length, n = matrix[0].length;
        int[][] up = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                up[i][j] = matrix[i][j] - '0';
                up[i][j] += i > 0 && up[i - 1][j] > 0 && up[i][j] > 0 ? up[i - 1][j] : 0;
            }
        }

        int res = 0;
        for (int i = 0; i < m; i++) {
            res = Math.max(res, largestRectangleArea(up[i]));
        }
        return res;
    }

    public static void disMap(int[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                System.out.print("(" + i + "," + j + ") " + a[i][j] + " ");
            }
            System.out.println();
        }
    }

    //1444. 切披萨的方案数
    public int ways(String[] pizza, int k) {
        if (pizza.length == 0 || pizza[0].length() == 0) {
            return 0;
        }
        int m = pizza.length, n = pizza[0].length(), mod = 1000000007;
        //        apple[i][j]:以apple[i][j]为左上角的剩余矩形所包含的苹果数
        int[][] apples = new int[m + 1][n + 1];
        //        dp[k][i][j]:以apple[i][j]为左上角的剩余矩形分为k块的方案数
        int[][][] dp = new int[k + 1][m + 1][n + 1];

        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                apples[i][j] = apples[i + 1][j] + apples[i][j + 1] - apples[i + 1][j + 1] + (pizza[i].charAt(j) == 'A' ? 1 : 0);
//                在获取苹果数的同时为一份披萨的初始化，只要有苹果就能够分成1快 且只有一种分法
                dp[1][i][j] = apples[i][j] > 0 ? 1 : 0;
            }
        }
        disMap(apples);

//        ki从2开始，1已经初始化
        for (int ki = 2; ki <= k; ki++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {

//                    加上所有的横切的可能性，在row处切掉的所有情况都加上，因为只要row处能够切都算一个切法
                    for (int row = i + 1; row < m; row++) {
//                        apples[i][j] 只可能>= apples[row][j],等于的情况不能切，因为上面没有苹果
                        if (apples[i][j] > apples[row][j]) {
                            dp[ki][i][j] = (dp[ki][i][j] + dp[ki - 1][row][j]) % mod;
                        }
                    }
                    for (int col = j + 1; col < n; col++) {
                        if (apples[i][j] > apples[i][col]) {
                            dp[ki][i][j] = (dp[ki][i][j] + dp[ki - 1][i][col]) % mod;
                        }
                    }

                }
            }
        }

        return dp[k][0][0];


    }

    //    LCR 162. 数字 1 的个数
    public int digitOneInNumber(int num) {
        int len = String.valueOf(num).length();
        long res = 0;
        for (int i = 0; i < len; i++) {
//            123100
            long mod = (int) Math.pow(10, i);
            System.out.println("mod" + mod);
            long l = num / (mod * 10);
            long r = num % (mod * 10);
            System.out.println(l);
            System.out.println(r);
            long lsum = l * mod;
            long rsum = Math.min(mod, Math.max(0, r - mod + 1));
            System.out.println("lsum:" + lsum + " rsum" + rsum);
            res += lsum;
            res += rsum;
        }
        System.out.println(res);
        return (int) res;


    }

    //37. 解数独
    public void solveSudoku(char[][] board) {
        backtrace(board, 0, 0);

    }

    public boolean backtrace(char[][] board, int i, int j) {
        int m = 9, n = 9;
        if (j == n) {
            return backtrace(board, i + 1, 0);
        }
        if (i == m) {
            return true;
        }
        if (board[i][j] != '.') {
            return backtrace(board, i, j + 1);
        }
        for (char c = '1'; c <= '9'; c++) {
            if (valid(board, i, j, c)) {
                board[i][j] = c;
                if (backtrace(board, i, j + 1)) {
                    return true;
                }
                board[i][j] = '.';
            }
        }
        return false;

    }

    public boolean valid(char[][] board, int i, int j, char c) {
        for (int k = 0; k < 9; k++) {
            if (board[k][j] == c || board[i][k] == c || board[(i / 3) * 3 + k / 3][(j / 3) * 3 + k % 3] == c) {
                return false;
            }
        }
        return true;
    }

    //    36. 有效的数独
    public boolean isValidSudoku(char[][] board) {
        for (int k = 0; k < 9; k++) {
            if (board)
            if (board[k][j] == c || board[i][k] == c || board[(i / 3) * 3 + k / 3][(j / 3) * 3 + k % 3] == c) {
                return false;
            }
        }
    }

    //679. 24 点游戏
    static final int A = 0, M = 1, S = 2, D = 3;
    static final double E = 1e-6;
    static final double TARGET = 1e-6;

    public boolean judgePoint24(int[] cards) {
        List<Double> nums = Arrays.stream(cards).mapToDouble(Double::valueOf).boxed().collect(Collectors.toList());
//        System.out.println(nums);
        return backtrace(nums);
    }

    public boolean backtrace(List<Double> nums) {
//        整体思路：枚举所有情况回溯，每次抽取两个数，之后针对每种符号情况计算之后放进新的数组里面，进行回溯，直到符合条件为止
//        需要注意的 点：1.+*交换律需要去重 /号需要分母0的处理，此时是不能remove尾巴的 因为这种情况不会add新的进去nums2 2.在符号的循环里面回溯（递归+remove）
        if (nums == null || nums.size() == 0) {
            return false;
        }
        if (nums.size() == 1) {
            return nums.get(0) == TARGET;
//            return Math.abs(nums.get(0) - TARGET) < E;
        }
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    List<Double> nums2 = new ArrayList<>();
                    for (int k = 0; k < n; k++) {
//                        放出了ij之外的数
                        if (k != i && k != j) {
                            nums2.add(nums.get(k));
                        }
                    }
                    for (int k = 0; k < 4; k++) {
                        if (k < 2 && j < i) {
//                            +*交换律 去重
                            continue;
                        }
                        if (k == A) {
                            nums2.add(nums.get(i) + nums.get(j));
                        } else if (k == M) {
                            nums2.add(nums.get(i) * nums.get(j));
                        } else if (k == S) {
                            nums2.add(nums.get(i) - nums.get(j));
                        } else if (k == D) {
//                            if (Math.abs(nums.get(j)) < E) {
                            if (nums.get(j) == 0) {
                                continue;
                            } else {
                                nums2.add(nums.get(i) / nums.get(j));
                            }
                        }
                        if (backtrace(nums2)) {
                            return true;
                        }
                        nums2.remove(nums2.size() - 1);
                    }
                }
            }
        }
        return false;

    }

    public static void main(String[] args) {
//        String rexp = "^((\\d{2}(([02468][048])|([13579][26]))[\\-\\/\\s]?((((0?[13578])|(1[02]))[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])|(3[01])))|(((0?[469])|(11))[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])|(30)))|(0?2[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])))))|(\\d{2}(([02468][1235679])|([13579][01345789]))[\\-\\/\\s]?((((0?[13578])|(1[02]))[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])|(3[01])))|(((0?[469])|(11))[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])|(30)))|(0?2[\\-\\/\\s]?((0?[1-9])|(1[0-9])|(2[0-8]))))))";
////        SingleThreadScheduledEx
//        Pattern p = Pattern.compile("[123]");
//        Matcher mat = p.matcher("4");
//        System.out.println(mat.matches());
        int[] nums = new int[]{2, 1, 5, 6, 2, 3};
        HardTest2 ht = new HardTest2();
//        System.out.println(ht.digitOneInNumber(1410065408));
//        ht.solveNQueens(4);
//        System.out.println(ht.digitOneInNumber(123456));

    }
}

//297. 二叉树的序列化与反序列化 前序遍历做法
class Codec1 {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serialize(root, sb);
        return sb.toString();

    }

    public void serialize(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append("#,");
            return;
        }
        sb.append(root.val).append(",");
        serialize(root.left, sb);
        serialize(root.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> q = new ArrayDeque<>();
        for (String s : data.split(",")) {
            q.offerLast(s);
        }
        return deserialize(q);

    }

    public TreeNode deserialize(Deque<String> q) {
        if (q.isEmpty()) {
            return null;
        }
        if (q.peekFirst().equals("#")) {
            q.pollFirst();
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(q.pollFirst()));
        root.left = deserialize(q);
        root.right = deserialize(q);
        return root;
    }
}

//297. 二叉树的序列化与反序列化 层序遍历做法
class Codec2 {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode r = q.poll();
            if (r == null) {
                sb.append("null,");
                continue;
            }
            sb.append(r.val).append(",");
            q.offer(r.left);
            q.offer(r.right);
        }
        System.out.println(sb.toString());
        return sb.toString();
    }


    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.isEmpty()) {
            return null;
        }
        String[] strings = data.split(",");
        if (strings.length == 0) {
            return null;
        }
//        for (String string : strings) {
//            System.out.println(string);
//        }
        TreeNode root = new TreeNode(Integer.parseInt(strings[0])), p;
        Deque<TreeNode> q = new ArrayDeque<>();
        q.offerLast(root);
        for (int i = 1; i < strings.length; ) {
            p = q.pollFirst();
            String l = strings[i++];
            if (!Objects.equals(l, "null")) {
                q.offerLast(p.left = new TreeNode(Integer.parseInt(l)));
            } else {
                p.left = null;
            }
            String r = strings[i++];
            if (!Objects.equals(r, "null")) {
                q.offerLast(p.right = new TreeNode(Integer.parseInt(r)));
            } else {
                p.right = null;
            }
        }
        return root;
    }

}

//460. LFU 缓存
class LFUCache {
    //总结：三个hash结构+一个int，其中一个hash需要使用linkedHashset保证随机访问加快速删除
    Map<Integer, Integer> kv;//为了get能够o1
    Map<Integer, Integer> kf;//为了超过容量的时候删除能够o1
    Map<Integer, LinkedHashSet<Integer>> fks;//为了能够删除最早的数据 就是为了保证找到最早的节点和删除最早的节点都是o1
    int cap;
    int minf;

    public LFUCache(int capacity) {
        this.cap = capacity;
        this.minf = 0;
        kv = new HashMap<>();
        kf = new HashMap<>();
        fks = new HashMap<>();
    }

    public int get(int key) {
        if (kv.containsKey(key)) {
            int oldf = kf.get(key);
            fks.get(oldf).remove(key);
            fks.putIfAbsent(oldf + 1, new LinkedHashSet<>());
            fks.get(oldf + 1).add(key);
            kf.put(key, oldf + 1);
            if (fks.get(oldf).isEmpty() && minf == oldf) {
                fks.remove(oldf);
                minf++;
            }
//            System.out.println("kv:" + kv);
//            System.out.println("kf:" + kf);

            return kv.get(key);

        } else {
            return -1;
        }

    }

    public void put(int key, int value) {

        if (kv.containsKey(key)) {
            kv.put(key, value);
            get(key);
        } else {
            kv.put(key, value);
            kf.put(key, 1);
            fks.putIfAbsent(1, new LinkedHashSet<>());
            fks.get(1).add(key);
            if (kv.size() > cap) {
                Integer del = fks.get(minf).iterator().next();
                kv.remove(del);
                kf.remove(del);
                fks.get(minf).remove(del);

            }
            minf = 1;
        }
//        System.out.println("kv:" + kv);
//        System.out.println("kf:" + kf);


    }
}


class Solution {
    static final int TARGET = 24;
    static final double EPSILON = 1e-6;
    static final int ADD = 0, MULTIPLY = 1, SUBTRACT = 2, DIVIDE = 3;

    public boolean judgePoint24(int[] nums) {
        List<Double> list = new ArrayList<Double>();
        for (int num : nums) {
            list.add((double) num);
        }
        return solve(list);
    }

    public boolean solve(List<Double> list) {
        if (list.size() == 0) {
            return false;
        }
        if (list.size() == 1) {
            return Math.abs(list.get(0) - TARGET) < EPSILON;
        }
        int size = list.size();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    List<Double> list2 = new ArrayList<Double>();
                    for (int k = 0; k < size; k++) {
                        if (k != i && k != j) {
                            list2.add(list.get(k));
                        }
                    }
                    for (int k = 0; k < 4; k++) {
                        if (k < 2 && i > j) {
                            continue;
                        }
                        if (k == ADD) {
                            list2.add(list.get(i) + list.get(j));
                        } else if (k == MULTIPLY) {
                            list2.add(list.get(i) * list.get(j));
                        } else if (k == SUBTRACT) {
                            list2.add(list.get(i) - list.get(j));
                        } else if (k == DIVIDE) {
                            if (Math.abs(list.get(j)) < EPSILON) {
                                continue;
                            } else {
                                list2.add(list.get(i) / list.get(j));
                            }
                        }
                        if (solve(list2)) {
                            return true;
                        }
                        list2.remove(list2.size() - 1);
                    }
                }
            }
        }
        return false;
    }
}

