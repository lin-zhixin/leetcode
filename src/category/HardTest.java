package category;

import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

//hard练习
public class HardTest {


    public void dis(ListNode head) {
        Object
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
//        &&左右两边的顺序不能换     如果换了的话下面的if条件也需要换
        while ((t = t - 1) >= 0 && b != null) {
            b = b.next;
        }
        if (t >= 0) {
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

    //    407. 接雨水 II
    public int trapRainWater(int[][] heightMap) {
//        使用堆，从最外围把最外围的所有元素放进堆中 然后在最外围里面找一条高度最小的路线 在路线行进的过程中每个位置的上下左右当中的能够填满的地方都填满，
//        同时把未访问的上下左右放进最小堆中
        if (heightMap.length < 3 || heightMap[0].length < 3) {
            return 0;
        }
        PriorityQueue<int[]> heap = new PriorityQueue<>((o1, o2) -> o1[2] - o2[2]);
        int m = heightMap.length, n = heightMap[0].length;
        boolean[][] v = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    //System.out.println(i + "," + j + " :" + heightMap[i][j]);
                    heap.offer(new int[]{i, j, heightMap[i][j]});
                    v[i][j] = true;
                }
            }
        }
        System.out.println(heap);
        int res = 0;
        List<Pair<Integer, Integer>> move = new ArrayList<>();
        move.add(new Pair<>(-1, 0));
        move.add(new Pair<>(1, 0));
        move.add(new Pair<>(0, -1));
        move.add(new Pair<>(0, 1));
        while (!heap.isEmpty()) {
            int[] cur = heap.poll();
            System.out.println(cur[0] + "," + cur[1] + " :" + cur[2]);
            for (Pair<Integer, Integer> p : move) {
                int nx = cur[0] + p.getKey(), ny = cur[1] + p.getValue();
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !v[nx][ny]) {
                    res += cur[2] > heightMap[nx][ny] ? cur[2] - heightMap[nx][ny] : 0;
                    heap.offer(new int[]{nx, ny, Math.max(heightMap[nx][ny], cur[2])});
                    v[nx][ny] = true;
                }
            }
        }
        return res;

        // [1,4,3,1,3,2]
        // [3,2,1,3,2,4]
        //[2,3,3,2,3,1]

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

    //124. 二叉树中的最大路径和
    int maxPathSum = -999999;

    public int maxPathSum(TreeNode root) {
        maxPathSum2(root);
        return maxPathSum;
    }

    public int maxPathSum2(TreeNode root) {
        if (root == null) {
            return -999999;
        }
        int left = maxPathSum2(root.left);
        int right = maxPathSum2(root.right);
        PriorityQueue<Integer> t = new PriorityQueue<>((o1, o2) -> o2 - o1);
        t.offer(root.val);
        t.offer(left + root.val);
        t.offer(right + root.val);
        int res = t.peek();
        t.offer(left);
        t.offer(right);
        t.offer(left + right + root.val);
        maxPathSum = Math.max(maxPathSum, t.peek());
//        System.out.println(max);
        return res;
    }


    //135. 分发糖果
    public int candy2(int[] ratings) {
        int n = ratings.length, inc = 1, dec = 0, pre = 1, res = 1;
        for (int i = 1; i < n; i++) {
            if (ratings[i] >= ratings[i - 1]) {
                dec = 0;
                pre = ratings[i] == ratings[i - 1] ? 1 : pre + 1;
                res += pre;
                inc = pre;
            } else {
//                降序的情况实际上就是从1开始加，反过来看  特殊情况就是dec == inc 降序序列长度等于上一个升序的时候要加一，因为降序的第一个值不能和升序最后一个相等 违背规则
                dec++;
                if (dec == inc) {
                    dec++;
                }
                res += dec;
                pre = 1;
            }
        }

        return res;

    }

    public int candy(int[] ratings) {
        int n = ratings.length, sum = 0;
        int[] res = new int[n];
        Arrays.fill(res, 1);
        for (int i = 1; i < n; i++) {
            if (ratings[i - 1] < ratings[i]) {
                res[i] = res[i - 1] + 1;
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && res[i] <= res[i + 1]) {
                res[i] = Math.max(res[i + 1] + 1, res[i] + 1);
            }
        }
//        MyUtile.dis(res);

        return Arrays.stream(res).sum();
    }

    //    51. N 皇后
    public List<List<String>> solveNQueens(int n) {
        int[][] map = new int[n][n];
        List<List<Pair<Integer, Integer>>> res = new ArrayList<>();
        solveNQueens(n, 0, new ArrayList<>(), res);
        System.out.println(res);
        return null;
    }

    public void solveNQueens(int n, int row, List<Pair<Integer, Integer>> points, List<List<Pair<Integer, Integer>>> res) {
        if (row == n) {
            res.add(new ArrayList<>(points));
        }
        for (int i = 0; i < n; i++) {
            if (isvalid(points, row, i)) {
                points.add(new Pair<>(row, i));
                solveNQueens(n, row + 1, points, res);
                points.remove(points.size() - 1);
            }
        }
    }

    public boolean isvalid(List<Pair<Integer, Integer>> points, int i, int j) {
        return points.stream().noneMatch(e -> i == e.getKey() || j == e.getValue() || Math.abs(i - e.getKey()) == Math.abs(j - e.getValue()));
    }


    //44. 通配符匹配
    public boolean isMatch2(String s, String p) {
        int[][] memo = new int[s.length()][p.length()];
        for (int i = 0; i < s.length(); i++) {
            Arrays.fill(memo[i], -1);
        }
        return isMatch2(s, p, 0, 0, memo);
    }

    public boolean isMatch2(String s, String p, int i, int j, int[][] memo) {
        int m = s.length(), n = p.length();
        if (j == n) {
            return i == m;
        }
        if (i == m) {
            for (int k = j; k < n; k++) {
                if (p.charAt(k) != '*') {
                    return false;
                }
            }
            return true;
        }
        if (memo[i][j] != -1) {
            return memo[i][j] == 1;
        }
        if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?') {
            return (memo[i][j] = isMatch2(s, p, i + 1, j + 1, memo) ? 1 : 0) == 1;
        } else {
            if (p.charAt(j) == '*') {
                return (memo[i][j] = (isMatch2(s, p, i, j + 1, memo) || isMatch2(s, p, i + 1, j, memo)) ? 1 : 0) == 1;
            } else {
                return (memo[i][j] = 0) == 1;
            }
        }
    }

    //115. 不同的子序列
    int numDistinctCnt = 0;

    public int numDistinct(String s, String t) {
        List<List<Pair<Integer, Integer>>> res = new ArrayList<>();
        int m = s.length(), n = t.length();

        int[][] memo = new int[m][n];
        for (int i = 0; i < m; i++) {
            Arrays.fill(memo[i], -1);
        }
//        numDistinctCnt = 0;
//        numDistinct(s, t, 0, 0, new LinkedList<>(), res, memo);
//        System.out.println(res);
//        return numDistinctCnt;
//        System.out.println(numDistinct(s, t, 0, 0, new LinkedList<>(), res, dp));
        return numDistinct(s, t, 0, 0, new LinkedList<>(), res, memo);
    }

    public int numDistinct(String s, String t, int i, int j, Deque<Integer> q, List<List<Pair<Integer, Integer>>> res, int[][] memo) {

        int m = s.length(), n = t.length();
        if (j == n) {
//            System.out.println(q);
            return 1;
        }
        if (i == m) {
            if (j < n) {
                return 0;
            }
        }
        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        int r = 0;
        if (s.charAt(i) == t.charAt(j)) {
            r += (numDistinct(s, t, i + 1, j + 1, q, res, memo) + numDistinct(s, t, i + 1, j, q, res, memo));
        } else {
            r += numDistinct(s, t, i + 1, j, q, res, memo);
        }
        return memo[i][j] = r;
    }


    public int lengthOfLIS(int[] nums) {
        int n = nums.length, piles = 0;
        int[] top = new int[n];
        for (int i = 0; i < n; i++) {
            int id = lowerBound(top, 0, piles - 1, nums[i]);
            System.out.println(id);
            if (id == piles) {
                piles++;
            }
            top[id] = nums[i];
        }
        return piles;

    }

    public int lowerBound(int[] nums, int l, int r, int target) {
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (target == nums[mid]) {
                r = mid - 1;
            } else if (target < nums[mid]) {
                r = mid - 1;
            } else if (target > nums[mid]) {
                l = mid + 1;
            }
        }

        return l;

    }


    //354.俄罗斯套娃信封问题
    public int maxEnvelopes(int[][] envelopes) {
        int n = envelopes.length, res = 0;
        Arrays.sort(envelopes, (o1, o2) -> o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0]);
        int[] nums = new int[n], dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 0; i < n; i++) {
            nums[i] = envelopes[i][1];
        }
        return lengthOfLIS(nums);
//        dp超时版本：
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < i; j++) {
//                if (nums[j] < nums[i]) {
//                    dp[i] = Math.max(dp[i], dp[j] + 1);
//                }
//            }
//            res = Math.max(res, dp[i]);
//        }
//        return res;


    }

    //57. 插入区间
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int n = intervals.length, i;
        int[][] ints = new int[n + 1][n + 1];
        for (i = 0; i < n; i++) {
            ints[i][0] = intervals[i][0];
            ints[i][1] = intervals[i][1];
        }
        ints[i][0] = newInterval[0];
        ints[i][1] = newInterval[1];
        Arrays.sort(intervals, (o1, o2) -> o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0]);
//        int[][] res=new int[n+1][n+1];
        int l = ints[0][0], r = ints[0][1];
        List<int[]> res = new ArrayList<>();
        for (int j = 0; j < ints.length; j++) {
            if (ints[i][0] <= r && r <= ints[i][1]) {
                r = ints[i][1];
            } else if (r < ints[i][0]) {
                res.add(new int[]{l, r});
                l = ints[i][0];
                r = ints[i][1];
            }
            System.out.println(l + " :" + r);
        }
        res.add(new int[]{l, r});
        System.out.println(res);
        return res.toArray(new int[res.size()][]);
    }

    //154. 寻找旋转排序数组中的最小值 II
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1, n = nums.length;

        while (l <= r) {
            int mid = l + (r - l) / 2;
//            用左边时候需要添加是否已经是单调的判断，否则会出界 原因：对于l来说一旦出现单调说明l一定是最小值，
//            因为正常l一定是大于等于r的 l的移动要么是mid+1要么++，两种情况不会产生跨越最小值的情况：
            if (nums[l] < nums[r]) {
                return nums[l];
            }
            if (nums[mid] < nums[l]) {
                r = mid;
            } else if (nums[mid] > nums[l]) {
                l = mid + 1;
            } else if (nums[mid] == nums[l]) {
                l++;
            }
        }
        return nums[l - 1];

    }


    //1444. 切披萨的方案数
    public int ways(String[] pizza, int k) {
        int m = pizza.length, n = pizza[0].length(), mod = 1000000007;
//        apple[i][j]:以apple[i][j]为左上角的剩余矩形所包含的苹果数
        int[][] apple = new int[m + 1][n + 1];
//        dp[k][i][j]:以apple[i][j]为左上角的剩余矩形分为k块的方案数
        int[][][] dp = new int[k + 1][m + 1][n + 1];

        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                apple[i][j] = apple[i + 1][j] + apple[i][j + 1] - apple[i + 1][j + 1] + (pizza[i].charAt(j) == 'A' ? 1 : 0);
                dp[1][i][j] = apple[i][j] > 0 ? 1 : 0;
            }
        }

        for (int i = 2; i <= k; i++) {
            for (int j = 0; j < m; j++) {
                for (int l = 0; l < n; l++) {

                    for (int row = j + 1; row < m; row++) {
                        if (apple[j][l] > apple[row][l]) {
                            dp[i][j][l] = (dp[i][j][l] + dp[i - 1][row][l]) % mod;
                        }
                    }
                    for (int col = l + 1; col < n; col++) {
                        if (apple[j][l] > apple[j][col]) {
                            dp[i][j][l] = (dp[i][j][l] + dp[i - 1][j][col]) % mod;
                        }
                    }
                }
            }

        }
        return dp[k][0][0];

    }

    //149. 直线上最多的点数
    public int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    public int maxPoints(int[][] points) {
        int n = points.length, res = 0;
        if (n == 1 || n == 2) {
            return n;
        }

        for (int i = 0; i < n; i++) {
            if (res > n / 2 || res >= n - i) {
                return res;
            }
            Map<Pair<Integer, Integer>, Integer> cnt = new HashMap<>();
            for (int j = i + 1; j < n; j++) {
                int dy = points[i][1] - points[j][1];
                int dx = points[i][0] - points[j][0];

                if (dx == 0) {
                    dy = 1;
                } else if (dy == 0) {
                    dx = 1;
                } else {
                    if (dx < 0) {
                        dx = -dx;
                        dy = -dy;
                    }
                    int gcd = gcd(Math.abs(dx), Math.abs(dy));
                    dx /= gcd;
                    dy /= gcd;
                }
                Pair<Integer, Integer> key = new Pair<>(dx, dy);
                cnt.put(key, cnt.getOrDefault(key, 1) + 1);
                res = Math.max(res, cnt.get(key));
            }
        }
        return res;


//        int n = points.length;
//        if (n <= 2) {
//            return n;
//        }
//        int ret = 0;
//        for (int i = 0; i < n; i++) {
//            if (ret >= n - i || ret > n / 2) {
//                break;
//            }
//            Map<Integer, Integer> map = new HashMap<Integer, Integer>();
//            for (int j = i + 1; j < n; j++) {
//                int x = points[i][0] - points[j][0];
//                int y = points[i][1] - points[j][1];
//                if (x == 0) {
//                    y = 1;
//                } else if (y == 0) {
//                    x = 1;
//                } else {
//                    if (y < 0) {
//                        x = -x;
//                        y = -y;
//                    }
//                    int gcdXY = gcd(Math.abs(x), Math.abs(y));
//                    x /= gcdXY;
//                    y /= gcdXY;
//                }
//                int key = y + x * 20001;
//                map.put(key, map.getOrDefault(key, 0) + 1);
//            }
//            int maxn = 0;
//            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
//                int num = entry.getValue();
//                maxn = Math.max(maxn, num + 1);
//            }
//            ret = Math.max(ret, maxn);
//        }
//        return ret;
    }

//    public int gcd(int a, int b) {
//        return b != 0 ? gcd(b, a % b) : a;
//    }
//    作者：力扣官方题解
//    链接：https://leetcode.cn/problems/max-points-on-a-line/solutions/842114/zhi-xian-shang-zui-duo-de-dian-shu-by-le-tq8f/
//    来源：力扣（LeetCode）
//    著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

    //496. 下一个更大元素 I
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> cnt = new HashMap<>();
        int n = nums2.length;
        Stack<Integer> stack = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums2[i]) {
                stack.pop();
            }
            cnt.put(nums2[i], stack.isEmpty() ? -1 : stack.peek());
            stack.push(nums2[i]);
        }
        for (int i = 0; i < nums1.length; i++) {
            nums1[i] = cnt.get(nums1[i]);
        }
        return nums1;

    }

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n];
        int[] right = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
//            >=表示只取低于当前h的下标
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i] && stack.pop() != null) ;
//            栈为空说明左边全是小于当前h的
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        stack.clear();
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i] && stack.pop() != null) ;
            //            栈为空说明右边全是小于当前h的
            right[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
//        MyUtile.dis(left);
//        MyUtile.dis(right);

        int res = 0;
        for (int i = 0; i < n; i++) {
//            right[i] - left[i] - 1：实际上是（r-1）-(l+1)+1
            res = Math.max(res, (right[i] - left[i] - 1) * heights[i]);
        }
        return res;
    }

    //85. 最大矩形
    public static void disMap(int[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                System.out.print("(" + i + "," + j + ") " + a[i][j] + " ");
            }
            System.out.println();
        }
    }

    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length, res = 0;
        if (m == 0) {
            return res;
        }
        int n = matrix[0].length;
        int[][] left = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j == 0) {
                    left[i][j] = matrix[i][j] - '0';
                    continue;
                }
                left[i][j] = matrix[i][j] == '1' ? left[i][j - 1] + 1 : 0;
            }
        }
        disMap(left);
        for (int j = 0; j < n; j++) {
            List<Integer> colNums = new ArrayList<>();
            for (int i = 0; i < m; i++) {
                colNums.add(left[i][j]);
            }
            int colmax = largestRectangleArea(colNums.stream().mapToInt(Integer::intValue).toArray());
            res = Math.max(res, colmax);
        }
        return res;

    }

    //28. 找出字符串中第一个匹配项的下标
    // Rabin-Karp 指纹字符串查找算法：https://mp.weixin.qq.com/s/HiEft8sW5QzHcM5mb5_V6g
    public int strStr(String haystack, String needle) {
//        整体思路就是通过待匹配的字符串的hash去确定是否在txt中可能存在相同串(hash相同初步确定可能相同 之后通过字符串的equals比较)
//        hash是生成就是根据类似字符串转int类型数字的表示方法确定的，就是当前数*进制+当前位的值；"1234"->123*10+4,同理推算256进制
        long n = haystack.length(), l = needle.length(), r = 256, q = Integer.MAX_VALUE, rl_1 = 1, patHash = 0;
        for (int i = 0; i < l - 1; i++) {
            rl_1 = (rl_1 * r) % q;
        }
        for (int i = 0; i < l; i++) {
            patHash = (patHash * r % q + needle.charAt(i)) % q;
        }
        int le = 0, ri = 0;
        long winHash = 0;
        while (ri < n) {
            winHash = (winHash * r % q + haystack.charAt(ri++)) % q;
            if (ri - le == l) {
                if (winHash == patHash && Objects.equals(haystack.substring(le, ri), needle)) {
                    return le;
                }
                // 因为 windowHash - (txt[left] * RL) % Q 可能是负数
                // 所以额外再加一个 Q，保证 windowHash 不会是负数
                winHash = (winHash - (haystack.charAt(le++) * rl_1) % q + q) % q;
            }
        }
        return -1;


//
//
////        // 位数
////        int L = pat.length();
////        // 进制（只考虑 ASCII 编码）
////        int R = 256;
////        // 取一个比较大的素数作为求模的除数
////        long Q = 1658598167;
////        // R^(L - 1) 的结果
////        long RL = 1;
////        for (int i = 1; i <= L - 1; i++) {
////            // 计算过程中不断求模，避免溢出
////            RL = (RL * R) % Q;
////        }
////        // 计算模式串的哈希值，时间 O(L)
////        long patHash = 0;
////        for (int i = 0; i < pat.length(); i++) {
////            patHash = (R * patHash + pat.charAt(i)) % Q;
////        }
////
////        // 滑动窗口中子字符串的哈希值
////        long windowHash = 0;
////
////        // 滑动窗口代码框架，时间 O(N)
////        int left = 0, right = 0;
////        while (right < txt.length()) {
////            // 扩大窗口，移入字符
////            windowHash = ((R * windowHash) % Q + txt.charAt(right)) % Q;
////            right++;
////
////            // 当子串的长度达到要求
////            if (right - left == L) {
////                // 根据哈希值判断是否匹配模式串
////                if (windowHash == patHash) {
////                    // 当前窗口中的子串哈希值等于模式串的哈希值
////                    // 还需进一步确认窗口子串是否真的和模式串相同，避免哈希冲突
////                    if (pat.equals(txt.substring(left, right))) {
////                        return left;
////                    }
////                }
////                // 缩小窗口，移出字符
////                windowHash = (windowHash - (txt.charAt(left) * RL) % Q + Q) % Q;
////                // X % Q == (X + Q) % Q 是一个模运算法则
////                // 因为 windowHash - (txt[left] * RL) % Q 可能是负数
////                // 所以额外再加一个 Q，保证 windowHash 不会是负数
////
////                left++;
////            }
////        }
////        // 没有找到模式串
////        return -1;
    }

    public int mypow(int n, int k) {
        int res = 1, base = n;
        while (k != 0) {
            if (k % 2 == 1) {
                res *= base;
            }
            k /= 2;
            base *= base;
        }
        return res;
    }

    //1044. 最长重复子串
//    先用187题目的 字符串哈希 + 前缀和解法先解出来吧，hard题目真的难。
    class Solution {
        long P = 31;
        long[] h;
        long[] p;

        public String longestDupSubstring(String s) {
            h = new long[s.length() + 1];
            p = new long[s.length() + 1];
            p[0] = 1;
            for (int i = 1; i <= s.length(); i++) {
                h[i] = h[i - 1] * P + (long) s.charAt(i - 1);
                p[i] = p[i - 1] * P;
            }
            int left = 0;
            int right = s.length();
            while (left < right) {
                int mid = left + (right - left) / 2;
                String ans = findRepeatedDnaSequences(s, mid);
                if (ans == "") {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            return findRepeatedDnaSequences(s, left - 1);
        }

        public String findRepeatedDnaSequences(String s, int len) {
            if (len < 0) return "";
            Map<Long, Integer> map = new HashMap<>();
            for (int i = len; i <= s.length(); i++) {
                long v = h[i] - h[i - len] * p[len];
                if (map.containsKey(v)) {
                    if (map.get(v) == 1) {
                        return s.substring(i - len, i);
                    }
                    map.put(v, map.get(v) + 1);
                } else map.put(v, 1);
            }
            return "";
        }
    }

    public String longestDupSubstring(String s) {
        int n = s.length(), l = 1, r = n - 1, start = -1, len = 0;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            int ind = rabinKarp(s, mid);
            if (ind != -1) {
                l = mid + 1;
                start = ind;
                len = mid;
            } else {
                r = mid - 1;
            }
        }
        return start == -1 ? "" : s.substring(start, start + len);
    }

    public int rabinKarp(String s, int len) {
        long R = 256, Rl1 = 1, Q = Integer.MAX_VALUE, winhash = 0;
        for (int i = 0; i < len - 1; i++) {
            Rl1 = (Rl1 * R) % Q;
        }
        int n = s.length(), l = 0, r = 0;
//        Map<Long, Set<String>> map = new HashMap<>();
        Set<Long> set = new HashSet<>();
        Map<Long, HashSet<String>> map = new HashMap<>();
        while (r < n) {
            winhash = (winhash * R % Q + s.charAt(r++)) % Q;
            if (r - l == len) {
//                if (map.containsKey(winhash) && map.get(winhash).contains(s.substring(l, r))) {
                if (!set.add(winhash) && s.indexOf(s.substring(l, r)) != l && map.get(winhash).contains(s.substring(l, r))) {
                    return l;
                } else {
                    map.putIfAbsent(winhash, new LinkedHashSet<>());
                    map.get(winhash).add(s.substring(l, r));
                }
//                if (map.containsKey(winhash)) {
//                } else {
//                    map.putIfAbsent(winhash, new LinkedHashSet<>());
//                    map.get(winhash).add(s.substring(l, r));
//                }
                winhash = (winhash - (s.charAt(l++) * Rl1) % Q + Q) % Q;
            }
        }
        return -1;
    }
//    public String longestDupSubstring(String s) {
//        Random random = new Random();
//        // 生成两个进制
//        int a1 = random.nextInt(75) + 26;
//        int a2 = random.nextInt(75) + 26;
//        // 生成两个模
//        int mod1 = random.nextInt(Integer.MAX_VALUE - 1000000007 + 1) + 1000000007;
//        int mod2 = random.nextInt(Integer.MAX_VALUE - 1000000007 + 1) + 1000000007;
//        int n = s.length();
//        // 先对所有字符进行编码
//        int[] arr = new int[n];
//        for (int i = 0; i < n; ++i) {
//            arr[i] = s.charAt(i) - 'a';
//        }
//        // 二分查找的范围是[1, n-1]
//        int l = 1, r = n - 1;
//        int length = 0, start = -1;
//        while (l <= r) {
//            int m = l + (r - l + 1) / 2;
//            int idx = check(arr, m, a1, a2, mod1, mod2);
//            if (idx != -1) {
//                // 有重复子串，移动左边界
//                l = m + 1;
//                length = m;
//                start = idx;
//            } else {
//                // 无重复子串，移动右边界
//                r = m - 1;
//            }
//        }
//        return start != -1 ? s.substring(start, start + length) : "";
//    }

    public int check(int[] arr, int m, int a1, int a2, int mod1, int mod2) {
        int n = arr.length;
        long aL1 = pow(a1, m, mod1);
        long aL2 = pow(a2, m, mod2);
        long h1 = 0, h2 = 0;
        for (int i = 0; i < m; ++i) {
            h1 = (h1 * a1 % mod1 + arr[i]) % mod1;
            h2 = (h2 * a2 % mod2 + arr[i]) % mod2;
            if (h1 < 0) {
                h1 += mod1;
            }
            if (h2 < 0) {
                h2 += mod2;
            }
        }
        // 存储一个编码组合是否出现过
        Set<Long> seen = new HashSet<Long>();
        seen.add(h1 * mod2 + h2);
        for (int start = 1; start <= n - m; ++start) {
            h1 = (h1 * a1 % mod1 - arr[start - 1] * aL1 % mod1 + arr[start + m - 1]) % mod1;
            h2 = (h2 * a2 % mod2 - arr[start - 1] * aL2 % mod2 + arr[start + m - 1]) % mod2;
            if (h1 < 0) {
                h1 += mod1;
            }
            if (h2 < 0) {
                h2 += mod2;
            }

            long num = h1 * mod2 + h2;
            // 如果重复，则返回重复串的起点
            if (!seen.add(num)) {
                return start;
            }
        }
        // 没有重复，则返回-1
        return -1;
    }

    public long pow(int a, int m, int mod) {
        long ans = 1;
        long contribute = a;
        while (m > 0) {
            if (m % 2 == 1) {
                ans = ans * contribute % mod;
                if (ans < 0) {
                    ans += mod;
                }
            }
            contribute = contribute * contribute % mod;
            if (contribute < 0) {
                contribute += mod;
            }
            m /= 2;
        }
        return ans;
    }
//    作者：力扣官方题解
//    链接：https://leetcode.cn/problems/longest-duplicate-substring/solutions/1171003/zui-chang-zhong-fu-zi-chuan-by-leetcode-0i9rd/
//    来源：力扣（LeetCode）
//    著作权归作者所有.商业转载请联系作者获得授权，非商业转载请注明出处。

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


        int[] nums1 = new int[]{1, 1};
        int[] nums2 = new int[]{3, 4};
//        hardTest.findMedianSortedArrays(nums1, nums2);
//        hardTest.maxSlidingWindow(nums1, 4);
//        hardTest.minWindow("ADOBECODEBANC", "ABC");
//        hardTest.isValid("()");
//        System.out.println(hardTest.longestValidParentheses("()"));
//        System.out.println(hardTest.calculate(" 2-1 + 2 "));
//        System.out.println(hardTest.candy(nums1));
//        System.out.println(hardTest.numDistinct("babgbag", "bag"));
//        System.out.println(hardTest.mypow(2, 31));
        System.out.println(hardTest.longestDupSubstring("banana"));
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

//297. 二叉树的序列化与反序列化
class Codec {

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
        Deque<String> q = new LinkedList<>();
        for (String s : data.split(",")) {
            q.offerLast(s);
        }
        return deserialize(q);
    }

    public TreeNode deserialize(Deque<String> q) {
        if (q.isEmpty()) {
            return null;
        }
        if (Objects.equals(q.peekFirst(), "#")) {
            q.pollFirst();
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(q.pollFirst()));
        root.left = deserialize(q);
        root.right = deserialize(q);
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec ser = new Codec();
// Codec deser = new Codec();
// TreeNode ans = deser.deserialize(ser.serialize(root));


class LRUCache3 {
    DlinkNode2 head = new DlinkNode2(), tail = new DlinkNode2();
    Map<Integer, DlinkNode2> cache = new HashMap<>();
    int cap, size;

    public LRUCache3(int capacity) {
        cap = capacity;
        size = 0;
        head.next = tail;
        tail.pre = head;
    }

    public int get(int key) {
        if (!cache.containsKey(key)) {
            return -1;
        }
        return move2Head(cache.get(key)).val;
    }

    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            DlinkNode2 node = cache.get(key);
            node.val = value;
            move2Head(node);
        } else {
            DlinkNode2 node = new DlinkNode2(key, value);
            cache.put(key, addHead(node));
            size++;
            if (size > cap) {
                cache.remove(delete(tail.pre).key);
                size--;
            }
        }

    }

    public DlinkNode2 delete(DlinkNode2 node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
        return node;
    }

    public DlinkNode2 move2Head(DlinkNode2 node) {
        return addHead(delete(node));
    }

    public DlinkNode2 addHead(DlinkNode2 node) {
        node.pre = head;
        node.next = head.next;
        head.next.pre = node;
        head.next = node;
        return node;
    }
}

class DlinkNode2 {
    int key, val;
    DlinkNode2 pre, next;

    public DlinkNode2() {
    }

    public DlinkNode2(int key, int val) {
        this.key = key;
        this.val = val;
    }
}


//460. LFU 缓存 重点在数据结构的选择
class LFUCache {

    Map<Integer, Integer> kv;
    Map<Integer, Integer> kf;
    //    LinkedHashSet结合hash和链表的优点实现快速O(1)访问、根据时间有序存储(能够快速删除最久未使用的节点)
    Map<Integer, LinkedHashSet<Integer>> fk;
    int minf;
    int cap;

    public LFUCache(int capacity) {
        cap = capacity;
        minf = 0;
        kv = new HashMap<>();
        kf = new HashMap<>();
        fk = new HashMap<>();
    }

    public int get(int key) {
        if (!kv.containsKey(key)) {
            return -1;

        }
        System.out.println("k" + key);
        int oldf = kf.get(key);

        kf.put(key, oldf + 1);
        fk.get(oldf).remove(key);
        fk.putIfAbsent(oldf + 1, new LinkedHashSet<>());
        fk.get(oldf + 1).add(key);
        if (oldf == minf && fk.get(oldf).isEmpty()) {
            minf++;
            fk.remove(oldf);
        }
//        System.out.println("kv:" + kv);
//        System.out.println("kf:" + kf);
        return kv.get(key);
    }

    public void put(int key, int value) {
        if (kv.containsKey(key)) {
            kv.put(key, value);
            get(key);
        } else {
            kv.put(key, value);
            kf.put(key, 1);
            fk.putIfAbsent(1, new LinkedHashSet<>());
            fk.get(1).add(key);
            if (kv.size() > cap) {
                Integer del = fk.get(minf).iterator().next();
                fk.get(minf).remove(del);
                kv.remove(del);
                kf.remove(del);
            }
            minf = 1;
        }
//        System.out.println("kv:" + kv);
//        System.out.println("kf:" + kf);
    }
}