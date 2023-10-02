package category;

import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

//hard练习
public class HardTest {
    public void dis(ListNode head) {
        ListNode p = head;
        while (Objects.nonNull(p)) {
            System.out.print(p.val + " ");
            p = p.next;
        }
        System.out.println();
    }

    //    25. K 个一组翻转链表
//递归
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode a = head, b = head;
        int t = k;
        while (b != null && (t = t - 1) >= 0) {
            b = b.next;
        }
        if (t > 0) {
            return head;
        }
        ListNode newHead = reverse(a, b);
        a.next = reverseKGroup(b, k);
        return newHead;
    }

    //非递归
    public ListNode reverseKGroup2(ListNode head, int k) {
        ListNode slow = new ListNode(), res = slow, fast = head;
        slow.next = head;
        int i;
        while (fast != null) {
            for (i = 0; i < k && fast != null; i++) {
                fast = fast.next;
            }
//            if (i < k) {
//                break;
//            }

            slow.next = reverse(slow.next, fast);

            for (i = 0; i < k && slow != null; i++) {
                slow = slow.next;
            }
        }
        dis(res.next);
        return res.next;
    }

    public ListNode reverse(ListNode a, ListNode b) {
//        [)
        ListNode pre = b;
        while (a != b) {
            ListNode next = a.next;
            a.next = pre;
            pre = a;
            a = next;
        }
        return pre;
    }

    //23. 合并 K 个升序链表
//    归并做法 knlogk
    public ListNode mergeKLists(ListNode[] lists) {
        return mergeKLists(lists, 0, lists.length - 1);
    }

    public ListNode mergeKLists(ListNode[] lists, int l, int r) {
        if (l < r) {
            int mid = (l + r) / 2;
            ListNode left = mergeKLists(lists, l, mid);
            ListNode right = mergeKLists(lists, mid + 1, r);
            return merge(left, right);
        }
        return l == r ? lists[l] : null;
    }

    public ListNode merge(ListNode a, ListNode b) {
        ListNode pa = a, pb = b, p3 = new ListNode(), res = p3;
        while (pa != null && pb != null) {
            if (pa.val <= pb.val) {
                p3 = (p3.next = new ListNode(pa.val));
                pa = pa.next;
            } else {
                p3 = (p3.next = new ListNode(pb.val));
                pb = pb.next;
            }
        }
        while (pa != null) {
            p3 = (p3.next = new ListNode(pa.val));
            pa = pa.next;
        }
        while (pb != null) {
            p3 = (p3.next = new ListNode(pb.val));
            pb = pb.next;
        }
        return res.next;

    }

    //    堆做法 knlogk
    public ListNode mergeKLists2(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> heap = new PriorityQueue<>(((o1, o2) -> o1.val - o2.val));
        int n = lists.length;
        for (int i = 0; i < n; i++) {
            if (lists[i] != null) {
                heap.offer(lists[i]);
            }
        }
        ListNode p3 = new ListNode(), res = p3;
        while (!heap.isEmpty()) {
//            System.out.println(heap.peek().val);
            ListNode peek = heap.poll();
            p3 = (p3.next = new ListNode(peek.val));
            if (peek.next != null) {
                heap.offer(peek.next);
            }
        }
        return res.next;
    }

    //42. 接雨水
    public int trap(int[] height) {
        int l = 0, r = height.length - 1, lmax = height[l], rmax = height[r], res = 0;
        while (l < r) {
            lmax = Math.max(lmax, height[l]);
            rmax = Math.max(rmax, height[r]);

            if (lmax <= rmax) {
                res += lmax - height[l++];
            } else {
                res += rmax - height[r--];
            }
        }
        return res;


    }

    //11.盛最多水的容器
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1, res = 0;
        while (l < r) {
            if (height[l] <= height[r]) {
                res = Math.max(res, height[l] * (r - l));
                l++;
            } else {
                res = Math.max(res, height[r] * (r - l));
                r--;
            }
        }
        return res;


    }

    //    72. 编辑距离
//    递归
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            Arrays.fill(memo[i], -1);
        }
        return minDistance(word1, 0, word2, 0, memo);
    }

    public int minDistance(String word1, int i, String word2, int j, int[][] memo) {
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
            return memo[i][j] = minDistance(word1, i + 1, word2, j + 1, memo);
        }
        return memo[i][j] = Math.min(Math.min(minDistance(word1, i + 1, word2, j, memo), minDistance(word1, i, word2, j + 1, memo)), minDistance(word1, i + 1, word2, j + 1, memo)) + 1;
    }

    // dp
    public int minDistance2(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0) {
                    dp[i][j] = j;
                    continue;
                }
                if (j == 0) {
                    dp[i][j] = i;
                    continue;
                }

                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
//                    dp[i][j - 1]:i后面插入一个
//                    dp[i - 1][j - 1]：替换
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];
    }


    //4. 寻找两个正序数组的中位
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length, k = (m + n) / 2;
        if ((m + n) % 2 == 1) {
            return findMedianSortedArrays(nums1, nums2, k + 1);
        } else {
            return (findMedianSortedArrays(nums1, nums2, k) + findMedianSortedArrays(nums1, nums2, k + 1)) / 2.0;
        }
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2, int k) {
        int m = nums1.length, n = nums2.length;
        int i = 0, j = 0;
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

            int half = k / 2;
            int newi = Math.min(i + half, m) - 1;
            int newj = Math.min(j + half, n) - 1;
            int p1 = nums1[newi], p2 = nums2[newj];
            if (p1 <= p2) {
                k -= newi - i + 1;
                i = newi + 1;
            } else {
                k -= newj - j + 1;
                j = newj + 1;
            }
        }

    }

    //239. 滑动窗口最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> q = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        int n = nums.length, l = 0, r = 0;
        while (r < n) {
            while (!q.isEmpty() && q.peekLast() < nums[r]) {
                q.pollLast();
            }
            q.offerLast(nums[r]);

            if (r >= k - 1) {
                res.add(q.peekFirst());
                if (q.peekFirst() == nums[l]) {
                    q.pollFirst();
                }
                l++;
            }
            r++;

        }
        System.out.println(res);
        return res.stream().mapToInt(Integer::intValue).toArray();

    }

    //41. 缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] < 0) {
                nums[i] = n + 1;
            }
        }

        for (int i = 0; i < n; i++) {
            if (Math.abs(nums[i]) <= n) {
                nums[Math.abs(nums[i]) - 1] *= nums[Math.abs(nums[i]) - 1] > 0 ? -1 : 1;
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;

    }

    //76. 最小覆盖子串
    public String minWindow(String s, String t) {
        Map<Character, Integer> win = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        int m = s.length(), n = t.length(), l = 0, r = 0, valid = 0, resl = 0, resr = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            need.put(t.charAt(i), need.getOrDefault(t.charAt(i), 0) + 1);
        }
        while (r < m) {
            Character cr = s.charAt(r++);
            if (need.containsKey(cr)) {
                win.put(cr, win.getOrDefault(cr, 0) + 1);
                if (Objects.equals(win.get(cr), need.get(cr))) {
                    valid++;
                }
                if (valid == need.size()) {
                    while (valid == need.size()) {
                        Character cl = s.charAt(l++);
                        if (need.containsKey(cl)) {
                            if (Objects.equals(win.get(cl), need.get(cl))) {
                                valid--;
                                if (r - l + 1 < resr - resl) {
                                    resl = l - 1;
                                    resr = r;
                                    System.out.println(s.substring(resl, resr));
                                }
                            }
                            win.compute(cl, (k, v) -> v - 1);
                        }
                    }
                }
            }

        }
        return resr == Integer.MAX_VALUE ? "" : s.substring(resl, resr);
    }

//20. 有效的括号

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            } else {
                if (c == ')') {
                    if (stack.isEmpty() || !stack.pop().equals('(')) {
                        return false;
                    }
                } else if (c == ']') {
                    if (stack.isEmpty() || !stack.pop().equals('[')) {
                        return false;
                    }
                } else if (c == '}') {
                    if (stack.isEmpty() || !stack.pop().equals('{')) {
                        return false;
                    }
                }
            }
        }
        return stack.isEmpty();
    }

    //22. 括号生成
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        generateParenthesis(n, n, new LinkedList<>(), res);
        return res;
    }

    public void generateParenthesis(int l, int r, Deque<Character> q, List<String> res) {

        if (r == 0) {
            StringBuilder sb = new StringBuilder();
            q.forEach(sb::append);
            res.add(sb.toString());
            return;
        }
        if (l > 0) {
            q.offerLast('(');
            generateParenthesis(l - 1, r, q, res);
            q.pollLast();
        }
        if (r > l) {
            q.offerLast(')');
            generateParenthesis(l, r - 1, q, res);
            q.pollLast();
        }
    }

    //921. 使括号有效的最少添加
    public int minAddToMakeValid(String s) {
        int ln = 0, rn = 0, res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                rn++;
            } else {
                rn--;
                if (rn < 0) {
                    res++;
                    rn = 0;
                }
            }
        }
        return rn > 0 ? res + rn : res;

    }


    //    1541. 平衡括号字符串的最少插入次数
    public int minInsertions(String s) {

//        (()

        int rn = 0, res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                rn += 2;
//                处理两个右括号要连续 中间不能被（ 断掉， rn奇数说明前面少了个右括号
                if (rn % 2 == 1) {
                    res++;
                    rn--;
                }
            } else {
                rn--;
                if (rn == -1) {
                    res++;
                    rn = 1;
                }
            }
        }
        return res + rn;
//)()())
//  -1 1 0 1 0 -1
    }

    //32. 最长有效括号
    public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
//        以s[i]结尾的最长有效字符串长度
        int[] dp = new int[s.length()];
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(i);
                dp[i] = 0;
            } else {
                if (!stack.isEmpty()) {
                    int leftind = stack.pop();
                    int len = (i - leftind + 1) + (leftind == 0 ? 0 : dp[leftind - 1]);
                    dp[i] = len;
                } else {
                    dp[i] = 0;
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
//参考：
//    public int longestValidParentheses(String s) {
//        Stack<Integer> stk = new Stack<>();
//        // dp[i] 的定义：记录以 s[i-1] 结尾的最长合法括号子串长度
//        int[] dp = new int[s.length() + 1];
//        for (int i = 0; i < s.length(); i++) {
//            if (s.charAt(i) == '(') {
//                // 遇到左括号，记录索引
//                stk.push(i);
//                // 左括号不可能是合法括号子串的结尾
//                dp[i + 1] = 0;
//            } else {
//                // 遇到右括号
//                if (!stk.isEmpty()) {
//                    // 配对的左括号对应索引
//                    int leftIndex = stk.pop();
//                    // 以这个右括号结尾的最长子串长度
//                    int len = 1 + i - leftIndex + dp[leftIndex];
//                    dp[i + 1] = len;
//                } else {
//                    // 没有配对的左括号
//                    dp[i + 1] = 0;
//                }
//            }
//        }
//        // 计算最长子串的长度
//        int res = 0;
//        for (int i = 0; i < dp.length; i++) {
//            res = Math.max(res, dp[i]);
//        }
//        return res;
//    }
    }

    //224. 基本计算器 递归分治
    public int calculate(String s) {
        Deque<Character> q = new LinkedList<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                continue;
            }
            q.offerLast(s.charAt(i));
        }
        return calculate(q);
    }

    //    获取单个括号内的结果
    public int calculate(Deque<Character> q) {
        int num = 0;
        char sign = '+';//num的前一个符号
        Stack<Integer> stack = new Stack<>();
        while (!q.isEmpty()) {
            char c = q.pop();
            if (Character.isDigit(c)) {
                num = num * 10 + (c - '0');
            }
            if (c == '(') {
                num = calculate(q);
            }


            if (!Character.isDigit(c) || q.isEmpty()) {
                if (sign == '+') {
                    stack.push(num);
                } else if (sign == '-') {
                    stack.push(-num);
                } else if (sign == '*') {
                    stack.push(stack.pop() * num);
                } else if (sign == '/') {
                    stack.push(stack.pop() / num);
                }
                if (c == ')') {
                    break;
                }
                sign = c;
                num = 0;
            }
        }
        return stack.stream().mapToInt(Integer::intValue).sum();
    }


    //123. 买卖股票的最佳时机 III
    public int maxProfit(int[] prices) {
//        k:最大交易上限
        int n = prices.length, k = 2;
        int[][][] dp = new int[n + 1][k + 1][2];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= k; j++) {
                if (i == 0) {
                    dp[0][j][0] = 0;
                    dp[0][j][1] = Integer.MIN_VALUE;
                    continue;
                }
                if (j == 0) {
                    dp[i][0][0] = 0;
                    dp[i][0][1] = Integer.MIN_VALUE;
                    continue;
                }
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i - 1]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i - 1]);
            }
        }
        return dp[n][k][0];

    }

    public int findKthNumber(int n, int k) {
        k--;//减一个是因为下面的cur能够直接作为结果返回，因为每次while结束之后cur都是下一个数
        int cur = 1;
        while (k > 0) {
            int cn = getChildNum(cur, n);
            if (cn <= k) {
                k -= cn;
                cur++;
            } else {
                cur *= 10;
                k--;
            }
        }
        return cur;
    }

    //获取以cur为根节点的的子树节点数（包括自己）
    public int getChildNum(int cur, int n) {
        int first = cur, last = cur, num = 0;
        while (first <= n) {
            num += (Math.min(last, n) - first + 1);
            first *= 10;
            last = last * 10 + 9;
        }
        return num;
    }

    public boolean preorderK(TrieNode2 root, int k, Deque<Integer> path, List<Integer> res, Boolean find) {
        if (root != null && !find) {
            path.offerLast(root.val);

            if (root.end) {
                k--;
            }
            if (k == 0) {
                int num = 0;
                for (Integer e : path) {
                    num = (num * 10) + e;
                }
                res.add(num);
//                find=Boolean.TRUE;
                System.out.println("end: " + res);
                return true;
            }
//            path.offerLast(root.val);
            for (int i = 0; i < root.child.length; i++) {
                if (root.realChildNum + (root.end ? 1 : 0) < k) {
                    k -= root.realChildNum + (root.end ? 1 : 0);
                    continue;
                }
                if (preorderK(root.child[i], k, path, res, find)) {
                    return true;
                }
            }
            path.pollLast();

        }
        return false;
    }

//    public int preorder(TrieNode2 root, int k, Deque<Integer> path, List<Integer> preorder) {
//        if (root != null) {
//            if (root.realChildNum + (root.end ? 1 : 0) <= k) {
//                if (k == 1 && root.end) {
//                    int num = 0;
//                    for (Integer e : path) {
//                        num = (num * 10) + e;
//                    }
//                    preorder.add(num);
//                    if (preorder.size() == k) {
//                        return num;
//                    }
//                }
//
//                for (int i = 0; i < root.child.length; i++) {
//                    preorder(root.child[i], k, path, preorder);
//                }
//                return preorder(root)
//            }
//            return preorder(ro);
//        }
//        path.offerLast(root.val);
//        if (root.end) {
//            int num = 0;
//            for (Integer e : path) {
//                num = (num * 10) + e;
//            }
//            preorder.add(num);
//            if (preorder.size() == k) {
//                return num;
//            }
//        }
//        for (int i = 0; i < root.child.length; i++) {
//            preorder(root.child[i], k, path, preorder);
//        }
//        path.pollLast();
//
//        return 0;
//    }

    //912. 排序数组
    public int[] sortArray(int[] nums) {
        mergesort(nums, 0, nums.length - 1);
        return nums;
    }

    public void mergesort(int[] nums, int l, int r) {
        if (l < r) {
            int mid = l + (r - l) / 2;
            mergesort(nums, l, mid);
            mergesort(nums, mid + 1, r);
            merge(nums, l, mid, r);
        }
    }

    public void merge(int[] nums, int l, int mid, int r) {
        Deque<Integer> q = new LinkedList<>();
        int p1 = l, p2 = mid + 1;
        while (p1 <= mid && p2 <= r) {
            if (nums[p1] <= nums[p2]) {
                q.offerLast(nums[p1++]);
            } else {
                q.offerLast(nums[p2++]);
            }
        }
        while (p1 <= mid) {
            q.offerLast(nums[p1++]);
        }
        while (p2 <= r) {
            q.offerLast(nums[p2++]);
        }
        for (int i = l; i <= r; i++) {
            nums[i] = q.pollFirst();
        }
    }

    //    315. 计算右侧小于当前元素的个数
    int[] cnt;

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        cnt = new int[n];
        Pair<Integer, Integer>[] pnums = new Pair[n];
        for (int i = 0; i < n; i++) {
            pnums[i] = new Pair<>(nums[i], i);
        }
        pmergesort(pnums, 0, n - 1);
//        MyUtile.dis(cnt);
        return Arrays.stream(cnt).boxed().collect(Collectors.toList());
    }

    public void pmergesort(Pair<Integer, Integer>[] nums, int l, int r) {
        if (l < r) {
            int mid = l + (r - l) / 2;
            pmergesort(nums, l, mid);
            pmergesort(nums, mid + 1, r);
            pmerge(nums, l, mid, r);
        }
    }

    public void pmerge(Pair<Integer, Integer>[] nums, int l, int mid, int r) {
        Deque<Pair<Integer, Integer>> q = new LinkedList<>();
        int p1 = l, p2 = mid + 1;
        while (p1 <= mid && p2 <= r) {
            if (nums[p1].getKey() <= nums[p2].getKey()) {
                q.offerLast(nums[p1++]);
                cnt[q.peekLast().getValue()] += p2 - mid - 1;
            } else {
                q.offerLast(nums[p2++]);
            }
        }
        while (p1 <= mid) {
            q.offerLast(nums[p1++]);
            cnt[q.peekLast().getValue()] += p2 - mid - 1;
        }
        while (p2 <= r) {
            q.offerLast(nums[p2++]);
        }
        for (int i = l; i <= r; i++) {
            nums[i] = q.pollFirst();
        }
    }


    //    LCR 170. 交易逆序对的总数
    public int reversePairs(int[] record) {
        int n = record.length;
        cnt = new int[n];
        Pair<Integer, Integer>[] pnums = new Pair[n];
        for (int i = 0; i < n; i++) {
            pnums[i] = new Pair<>(record[i], i);
        }
        pmergesort(pnums, 0, n - 1);
//        MyUtile.dis(cnt);
        return Arrays.stream(cnt).sum();
    }

    //    493. 翻转对 (未完成)
    public int reversePairs2(int[] nums) {
        int n = nums.length;
        cnt = new int[n];
        Pair<Integer, Integer>[] pnums = new Pair[n];
        for (int i = 0; i < n; i++) {
            pnums[i] = new Pair<>(nums[i], i);
        }
        pmergesort2(pnums, 0, n - 1);
//        MyUtile.dis(cnt);
        return Arrays.stream(cnt).sum();
    }

    public void pmergesort2(Pair<Integer, Integer>[] nums, int l, int r) {
        if (l < r) {
            int mid = l + (r - l) / 2;
            pmergesort2(nums, l, mid);
            pmergesort2(nums, mid + 1, r);
            pmerge2(nums, l, mid, r);
        }
    }

    public void pmerge2(Pair<Integer, Integer>[] nums, int l, int mid, int r) {
        Deque<Pair<Integer, Integer>> q = new LinkedList<>();
        int p1 = l, p2 = mid + 1;
        while (p1 <= mid && p2 <= r) {
            if (nums[p1].getKey() <= nums[p2].getKey()) {
                q.offerLast(nums[p1++]);

                for (int i = mid + 1; i <= p2; i++) {
                    if (q.peekLast().getKey() > 2 * nums[i].getKey()) {
                        cnt[q.peekLast().getValue()]++;
                    }
                }
            } else {
                q.offerLast(nums[p2++]);
            }
        }
        while (p1 <= mid) {
            q.offerLast(nums[p1++]);
            for (int i = mid + 1; i <= p2; i++) {
                if (q.peekLast().getKey() > 2 * nums[i].getKey()) {
                    cnt[q.peekLast().getValue()]++;
                }
            }
        }
        while (p2 <= r) {
            q.offerLast(nums[p2++]);
        }
        for (int i = l; i <= r; i++) {
            nums[i] = q.pollFirst();
        }
    }


    //10. 正则表达式匹配
    public boolean isMatch(String s, String p) {
        return isMatch(s, p, 0, 0);
    }

    public boolean isMatch(String s, String p, int i, int j) {
        int m = s.length(), n = p.length();
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
        if (j == n) {
            return false;
        }
        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.') {
            if (j + 1 < n && p.charAt(j + 1) == '*') {
                return isMatch(s, p, i, j + 2) || isMatch(s, p, i + 1, j);
            } else {
                return isMatch(s, p, i + 1, j + 1);
            }
        } else {
            if (j + 1 < n && p.charAt(j + 1) == '*') {
                return isMatch(s, p, i, j + 2);
            } else {
                return false;
            }
        }
    }

    //329. 矩阵中的最长递增路径
    public int longestIncreasingPath(int[][] matrix) {
        List<Pair<Integer, Integer>> move = new ArrayList<>();
        move.add(new Pair<>(-1, 0));
        move.add(new Pair<>(1, 0));
        move.add(new Pair<>(0, -1));
        move.add(new Pair<>(0, 1));
        int m = matrix.length, n = matrix[0].length, res = 0;
        int[][] memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                res = Math.max(res, longestIncreasingPath(matrix, i, j, move, memo));
            }
        }
        return res;
    }

    public int longestIncreasingPath(int[][] matrix, int i, int j, List<Pair<Integer, Integer>> move, int[][] memo) {
        int m = matrix.length, n = matrix[0].length;
        if (i < 0 || i >= m || j < 0 || j >= n) {
            return 0;
        }
        if (memo[i][j] != 0) {
            return memo[i][j];
        }
        return memo[i][j] = move.stream().map(e -> {
            int newi = i + e.getKey(), newj = j + e.getValue();
            if ((newi >= 0 && newi < m && newj >= 0 && newj < n) && matrix[newi][newj] > matrix[i][j]) {
                return longestIncreasingPath(matrix, newi, newj, move, memo) + 1;
            }
            return 1;
        }).mapToInt(Integer::intValue).max().getAsInt();
    }

    public static void main(String[] args) {
        int[] ids = new int[]{1, 2, 3, 4, 5};
        ListNode dummyNode = new ListNode(-1);
        ListNode p = dummyNode;
        for (int e : ids) {
            p.next = new ListNode(e);
            p = p.next;
        }
        int[] ids2 = new int[]{1, 3, 4};
        ListNode dummyNode2 = new ListNode(-1);
        ListNode p2 = dummyNode2;
        for (int e : ids2) {
            p2.next = new ListNode(e);
            p2 = p2.next;
        }

        int[] ids3 = new int[]{2, 6};
        ListNode dummyNode3 = new ListNode(-1);
        ListNode p3 = dummyNode3;
        for (int e : ids3) {
            p3.next = new ListNode(e);
            p3 = p3.next;
        }

        HardTest hardTest = new HardTest();
//        System.out.println(hardTest.reverseKGroup2(dummyNode.next, 3));


        int[] nums1 = new int[]{2, 4, 3, 5, 1};
        int[] nums2 = new int[]{3, 4};
//        hardTest.findMedianSortedArrays(nums1, nums2);
//        hardTest.maxSlidingWindow(nums1, 4);
//        hardTest.minWindow("ADOBECODEBANC", "ABC");
//        hardTest.isValid("()");
//        System.out.println(hardTest.longestValidParentheses("()"));
//        System.out.println(hardTest.calculate(" 2-1 + 2 "));
        System.out.println(hardTest.reversePairs2(nums1));
    }
}

class TrieNode2 {
    int val;
    boolean end;
    int realChildNum = 0;
    TrieNode2[] child = new TrieNode2[10];

    public TrieNode2() {
    }

    public TrieNode2(int val) {
        this.val = val;
    }
}

class MedianFinder2 {
    PriorityQueue<Integer> left;
    PriorityQueue<Integer> right;

    public MedianFinder2() {
        left = new PriorityQueue<>(((o1, o2) -> o2 - o1));
        right = new PriorityQueue<>();
    }

    public void addNum(int num) {
        if (left.size() <= right.size()) {
            right.offer(num);
            left.offer(right.poll());
        } else {
            left.offer(num);
            right.offer(left.poll());
        }
    }

    public double findMedian() {
        if (left.size() == right.size()) {
            return (left.peek() + right.peek()) / 2.0;
        }
        return left.size() > right.size() ? left.peek() : right.peek();

    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */


