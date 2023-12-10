package interview_prepare;

import category.MyUtile;
import javafx.util.Pair;

import java.io.File;
import java.lang.management.BufferPoolMXBean;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Collectors;

public class MT {
    //    使用递归算法打印目录树
    public void printFile() {
        File f = new File("F:/for_work/java简历");
        printFile(f, 0);
    }

    public void printFile(File f, int level) {
        for (int i = 0; i < level; i++) {
            System.out.print("- ");
        }
        System.out.println(f.getName());
        if (f.isDirectory()) {
            File[] dic = f.listFiles();
            for (File file : dic) {
                printFile(file, level + 1);
            }
        }
    }

    //插入排序
    public void insertSort(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n - 1; i++) {
            int t = nums[i + 1], j;
            for (j = i + 1; j > 0 && nums[j - 1] > t; j--) {
                nums[j] = nums[j - 1];
            }
            nums[j] = t;
        }
    }

    //冒泡排序
    public void bsort(int[] nums) {
        boolean change = true;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (nums[j] > nums[j + 1]) {
                    MyUtile.swap(nums, j, j + 1);
                    change = true;
                }
                if (!change) {
                    break;
                }
            }
        }
    }

    //188. 买卖股票的最佳时机 IV
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        int[][][] dp = new int[n + 1][k + 1][2];

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= k; j++) {
                if (i == 0) {
                    dp[0][j][0] = 0;
                    dp[0][j][1] = -99999;
                    continue;
                }
                if (j == 0) {
                    dp[i][0][0] = 0;
                    dp[i][0][1] = -99999;
                    continue;
                }
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i - 1]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i - 1]);
            }
        }
        return dp[n][k][0];
    }

    //    122. 买卖股票的最佳时机 II
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n + 1][2];

        for (int i = 0; i <= n; i++) {
            if (i == 0) {
                dp[0][0] = 0;
                dp[0][1] = -99999;
                continue;
            }

            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i - 1]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i - 1]);
        }
        return dp[n][0];
    }


    //    5. 最长回文子串
    public String longestPalindrome(String s) {
        int n = s.length(), l = 0, max = 0;
        boolean[][] dp = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                dp[i][j] = i == j || s.charAt(i) == s.charAt(j) && j - i == 1 || s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1];
                if (dp[i][j] && (j - i + 1) > max) {
                    max = (j - i + 1);
                    l = i;
                }
            }
        }
        MyUtile.disMap(dp);
//        System.out.println(l + "," + r);
        return s.substring(l, l + max);

    }


    //95.不同的二叉搜索树 II(不熟练)
    public List<TreeNode> generateTrees(int n) {
        return generateTrees(1, n);
    }

    public List<TreeNode> generateTrees(int l, int r) {
        List<TreeNode> res = new ArrayList<>();
        if (l > r) {
            res.add(null);
            return res;
        }
        if (l == r) {
            res.add(new TreeNode(l));
            return res;
        }
        for (int root = l; root <= r; root++) {
            List<TreeNode> left = generateTrees(l, root - 1);
            List<TreeNode> right = generateTrees(root + 1, r);
            for (TreeNode ln : left) {
                for (TreeNode rn : right) {
                    TreeNode node = new TreeNode(root);
                    node.left = ln;
                    node.right = rn;
                    res.add(node);
                }
            }
        }
        return res;
    }

    //    96. 不同的二叉搜索树
    public int numTrees(int n) {
        int[][] memo = new int[n + 1][n + 1];
        for (int i = 0; i < memo.length; i++) {
            Arrays.fill(memo[i], -1);
        }

//        int[][] dp = new int[n + 1][n + 1];
//        dp[0][0]=1;
//        for (int i = 0; i < n; i++) {
//            for (int j = 1; j < n; j++) {
//                dp[i][i] += dp[i][j - 1] * dp[j + 1][i];
//            }
//
//        }

        return numTrees(1, n, memo);
    }

    public int numTrees(int l, int r, int[][] memo) {
        if (l > r) {
            return 1;
        }
        if (memo[l][r] != -1) {
            return memo[l][r];
        }
        int res = 0;
        for (int root = l; root <= r; root++) {
            int left = numTrees(l, root - 1, memo);
            int right = numTrees(root + 1, r, memo);
            res += left * right;
        }
        return memo[l][r] = res;
    }

    //105. 从前序与中序遍历序列构造二叉树
    public TreeNode buildTree0(int[] preorder, int[] inorder) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return buildTree0(map, inorder, preorder, 0, inorder.length - 1, 0, preorder.length - 1, null);
    }

    public TreeNode buildTree0(Map<Integer, Integer> map, int[] inorder, int[] preorder, int il, int ir, int pl, int pr, TreeNode root) {
        if (il < 0 || pl < 0 || il > ir || pl > pr) {
            return null;
        }
        root = new TreeNode(preorder[pl]);
        if (il == ir || pl == pr) {
            return root;
        }
        int ri = map.get(preorder[pl]), len = ri - il;
        root.left = buildTree0(map, inorder, preorder, il, ri - 1, pl + 1, pl + len, root.left);
        root.right = buildTree0(map, inorder, preorder, ri + 1, ir, pl + len + 1, pr, root.right);
        return root;
    }

    //106. 从中序与后序遍历序列构造二叉树
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return buildTree(map, inorder, postorder, 0, inorder.length - 1, 0, postorder.length - 1, null);
    }

    public TreeNode buildTree(Map<Integer, Integer> map, int[] inorder, int[] postorder, int il, int ir, int pl, int pr, TreeNode root) {
        if (il < 0 || pl < 0 || il > ir || pl > pr) {
            return null;
        }
        root = new TreeNode(postorder[pr]);
        if (il == ir || pl == pr) {
            return new TreeNode(postorder[pr]);
        }
        int ri = map.get(postorder[pr]), len = ri - il;
        root.left = buildTree(map, inorder, postorder, il, ri - 1, pl, pl + len - 1, root.left);
        root.right = buildTree(map, inorder, postorder, ri + 1, ir, pl + len, pr - 1, root.right);
        return root;
    }

    //124. 二叉树中的最大路径和
    int maxPathSum = -99999;

    public int maxPathSum(TreeNode root) {
        maxPathSum2(root);
        return maxPathSum;

    }

    public int maxPathSum2(TreeNode root) {
        if (root == null) {
            return -99999;
        }
        int l = maxPathSum2(root.left);
        int r = maxPathSum2(root.right);
        PriorityQueue<Integer> heap = new PriorityQueue<>((o1, o2) -> o2 - o1);
        heap.offer(root.val);
        heap.offer(root.val + l);
        heap.offer(root.val + r);
        int res = heap.peek();
        heap.offer(l);
        heap.offer(r);
        heap.offer(root.val + l + r);
        maxPathSum = Math.max(maxPathSum, heap.peek());
        return res;
    }

    //404. 左叶子之和
    public int sumOfLeftLeaves(TreeNode root) {
        return sumOfLeftLeaves2(root);
    }

    public int sumOfLeftLeaves2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left != null && root.left.left == null && root.left.right == null) {
            return root.left.val + sumOfLeftLeaves2(root.right);
        }
        return sumOfLeftLeaves2(root.left) + sumOfLeftLeaves2(root.right);
    }

    //108. 将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1, null);
    }

    public TreeNode sortedArrayToBST(int[] nums, int l, int r, TreeNode root) {
        if (l <= r) {
            int mid = l + (r - l) / 2;
            root = new TreeNode(nums[mid]);
            root.left = sortedArrayToBST(nums, l, mid - 1, root.left);
            root.right = sortedArrayToBST(nums, mid + 1, r, root.right);
        }
        return root;
    }

    //94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        List<Integer> res = new ArrayList<>();
        TreeNode pre = null, cur = root;
        while (!stack.isEmpty() || cur != null) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            res.add(stack.peek().val);
            cur = stack.pop().right;
        }
        return res;
    }

    //    145. 二叉树的后序遍历
    public List<Integer> postorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        List<Integer> res = new ArrayList<>();
        TreeNode pre = null, cur = root;
        while (!stack.isEmpty() || cur != null) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            TreeNode peek = stack.peek();
            if (peek.right != null && peek.right != pre) {
                cur = peek.right;
            } else {
                res.add((pre = stack.pop()).val);
            }
        }
        return res;

    }


    //49. 字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
        int n = strs.length;
        Map<String, List<String>> map = new HashMap();
        for (int i = 0; i < n; i++) {
            String t = strs[i];
            char[] cnt = new char[26];
            for (int j = 0; j < t.length(); j++) {
                cnt[t.charAt(j) - 'a']++;
            }
            String s = new String(cnt);
//            StringBuilder sb = new StringBuilder();
//            for (int j = 0; j < cnt.length; j++) {
//                sb.append(cnt[j]);
//            }
            map.putIfAbsent(s, new ArrayList<>());
            map.get(s).add(strs[i]);
        }
        return new ArrayList<>(map.values());
    }

    //7. 整数反转
    public int reverse(int x) {
        long n = 0, sig = x > 0 ? 1 : -1;
        x = Math.abs(x);
        while (x > 0) {
            n *= 10;
            n += x % 10;
            x /= 10;
        }
//        long res = n * sig;
        if (n > Integer.MAX_VALUE || n < Integer.MIN_VALUE) {
            return 0;
        }
        return (int) ((int) n * sig);
    }

    //26. 删除有序数组中的重复项
    public int removeDuplicates(int[] nums) {
        int l = 0, r = 0, n = nums.length;
        while (r < n) {
            if (nums[l] == nums[r]) {
                r++;
            } else {
                nums[++l] = nums[r++];
            }
        }
        return l + 1;

    }

    //    80. 删除有序数组中的重复项 II
    public int removeDuplicates2(int[] nums) {
        int pre = Integer.MAX_VALUE, l = 0, r = 0, n = nums.length, cnt = 1;
        while (r < n) {
            if (pre == nums[r]) {
                if (cnt < 2) {
                    nums[l++] = pre = nums[r];
                }
                cnt++;
                r++;//放在if外面，无论如何都要++
            } else {
                nums[l++] = pre = nums[r++];
                cnt = 1;
            }
        }
//        MyUtile.dis(nums);
        return l;

    }

    //88. 合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int tail = m + n - 1, p1 = m - 1, p2 = n - 1;
        while (p1 >= 0 && p2 >= 0) {
            if (nums1[p1] >= nums2[p2]) {
                nums1[tail--] = nums1[p1--];
            } else {
                nums1[tail--] = nums2[p2--];
            }
        }
        while (p1 >= 0) {
            nums1[tail--] = nums1[p1--];
        }
        while (p2 >= 0) {
            nums1[tail--] = nums2[p2--];
        }


    }

    //    面试题 17.14. 最小K个数
    public int[] smallestK(int[] arr, int k) {
        change(arr);
        List<Integer> res = new ArrayList<>();
        smallestK2(arr, k, 0, arr.length - 1, res);
        System.out.println(res);
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

    public void smallestK2(int[] nums, int k, int l, int r, List<Integer> res) {
        if (l < r) {
            int p = part(nums, l, r);
            if (p == k) {

                res.addAll(Arrays.stream(nums).boxed().collect(Collectors.toList()).subList(0, k));
                System.out.println(res);
                return;
            } else if (p < k) {
                smallestK2(nums, k, p + 1, r, res);
            } else if (p > k) {
                smallestK2(nums, k, l, p - 1, res);
            }
        }
    }

    //    215. 数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        change(nums);
        return findKthLargest(nums, nums.length - k, 0, nums.length - 1);
    }

    public int findKthLargest(int[] nums, int k, int l, int r) {
        if (l < r) {
            int p = part(nums, l, r);
            if (p == k) {
                return nums[p];
            } else if (p < k) {
                return findKthLargest(nums, k, p + 1, r);
            } else if (p > k) {
                return findKthLargest(nums, k, l, p - 1);
            }
        }
        return nums[l];
    }

    //912. 排序数组
    public int[] sortArray(int[] nums) {
        change(nums);
        qsort(nums, 0, nums.length - 1);
        return nums;
    }

    public void change(int[] nums) {
        int n = nums.length;
        Random r = new Random();
        for (int i = 0; i < n; i++) {
            int j = i + r.nextInt(n - i);
            int t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;
        }
    }

    public void qsort(int[] nums, int l, int r) {
        if (l < r) {
            int p = part(nums, l, r);
            qsort(nums, l, p);
            qsort(nums, p + 1, r);
        }
    }

    public int part(int[] nums, int l, int r) {
        int p = nums[l];
        while (l < r) {
            while (l < r && nums[r] >= p) r--;
            nums[l] = nums[r];
            while (l < r && nums[l] <= p) l++;
            nums[r] = nums[l];
        }
        nums[l] = p;
        return l;

    }


    //34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int l = lowerBound(nums, target);
        int r = upperBound(nums, target);
        if (l >= nums.length || nums[l] != target) l = -1;
        if (r > nums.length || r <= 0 || nums[r - 1] != target) r = -1;
        else r--;
        return new int[]{l, r};
    }

    public int lowerBound(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (nums[m] == target) {
                r = m - 1;
            } else if (nums[m] < target) {
                l = m + 1;
            } else if (nums[m] > target) {
                r = m - 1;
            }
        }
        return l;
    }

    public int upperBound(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (nums[m] == target) {
                l = m + 1;
            } else if (nums[m] < target) {
                l = m + 1;
            } else if (nums[m] > target) {
                r = m - 1;
            }
        }
        return l;
    }

    //    ------------链表
    //82. 删除排序链表中的重复元素 II
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode nh = new ListNode(-1, head), p = nh;

        while (p.next != null && p.next.next != null) {
            if (p.next.val == p.next.next.val) {
                int x = p.next.val;
                while (p.next != null && p.next.val == x) {
                    p.next = p.next.next;
                }
            } else {
                p = p.next;
            }
        }
        return nh.next;
    }

    //    LCR 077. 排序链表(自底向上)
    public ListNode sortList(ListNode head) {
//        if (head == null) {
//            return null;
//        }
        int len = 0;
        ListNode p = head;
        while (p != null) {
            len++;
            p = p.next;
        }
        ListNode res = new ListNode(-1, head);
        for (int i = 1; i < len; i *= 2) {
            ListNode cur = res.next, pre = res;
            while (cur != null) {
                ListNode m1 = cur;
                for (int j = 1; j < i && cur.next != null; j++) {
                    cur = cur.next;
                }
                ListNode m2 = cur.next;
                cur.next = null;
                cur = m2;
                for (int j = 1; j < i && cur != null && cur.next != null; j++) {
                    cur = cur.next;
                }
                if (cur != null) {
                    ListNode next = cur.next;
                    cur.next = null;
                    cur = next;
                }
                pre.next = merge2(m1, m2);
                while (pre.next != null) {
                    pre = pre.next;
                }
            }
        }
        return res.next;
    }

    public ListNode merge2(ListNode a, ListNode b) {
        ListNode nh = new ListNode(), p = nh;
        while (a != null && b != null) {
            if (a.val <= b.val) {
                p.next = a;
                a = a.next;
                p = p.next;
                p.next = null;
            } else {
                p.next = b;
                b = b.next;
                p = p.next;
                p.next = null;
            }
        }
        if (a != null) {
            p.next = a;
        } else {
            p.next = b;
        }
        return nh.next;
    }


    //92. 反转链表 II
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode pre = null, s = head, f = head;
        int len = right - left;
        while ((len = len - 1) >= 0) {
            f = f.next;
        }
        left--;
        while ((left = left - 1) >= 0) {
            pre = s;
            s = s.next;
            f = f.next;
        }
        if (pre != null) {
            pre.next = reverse(s, f.next);
            return head;
        } else {
            return reverse(s, f.next);
        }
    }

    public ListNode reverse(ListNode a, ListNode b) {
        ListNode pre = b, p = a;
        while (p != b && p != null) {
            ListNode next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        return pre;

    }

    //2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode p1 = l1, p2 = l2, nh = new ListNode(), p3 = nh;
        int c = 0;
        while (p1 != null && p2 != null) {
            int t = p1.val + p2.val + c;
            c = t > 9 ? 1 : 0;
            p3.next = new ListNode(t % 10, null);

            p1 = p1.next;
            p2 = p2.next;
            p3 = p3.next;
        }
        while (p1 != null) {
            int t = p1.val + c;
            c = t > 9 ? 1 : 0;
            p3.next = new ListNode(t % 10, null);
            p1 = p1.next;
            p3 = p3.next;

        }
        while (p2 != null) {
            int t = p2.val + c;
            c = t > 9 ? 1 : 0;
            p3.next = new ListNode(t % 10, null);
            p2 = p2.next;
            p3 = p3.next;

        }
        if (c > 0) {
            p3.next = new ListNode(c, null);
        }
        return reverse(nh.next);

    }


    //21. 合并两个有序链表
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode nh = new ListNode(), p1 = list1, p2 = list2, p3 = nh;
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                nh.next = p1;
                p1 = p1.next;
            } else {
                nh.next = p2;
                p2 = p2.next;
            }
            nh = nh.next;
            nh.next = null;
        }
        while (p1 != null) {
            nh.next = p1;
            p1 = p1.next;
            nh = nh.next;
            nh.next = null;
        }
        while (p2 != null) {
            nh.next = p2;
            p2 = p2.next;
            nh = nh.next;
            nh.next = null;
        }
        return p3.next;

    }

    //    86. 分隔链表
    public ListNode partition(ListNode head, int x) {
        ListNode lh = new ListNode(), mh = new ListNode(), p1 = lh, more = mh;
        mh.next = head;
        while (more.next != null) {
            if (more.next.val <= x) {
                ListNode t = more.next;
                more.next = t.next;
                t.next = null;
                p1.next = t;
                p1 = p1.next;
            } else {
                more = more.next;
            }
        }
        p1.next = mh.next;
        head = lh.next;
        disListNode(head);
        return head;

    }

    //23. 合并 K 个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> heap = new PriorityQueue<>((o1, o2) -> o1.val - o2.val);
        int n = lists.length;
        for (int i = 0; i < n; i++) {
            if (lists[i] != null) {
                heap.offer(lists[i]);
            }
        }
        ListNode res = new ListNode(), p = res;
        while (!heap.isEmpty()) {
            res.next = heap.poll();
            if (res.next.next != null) {
                heap.offer(res.next.next);
            }
            res = res.next;
            res.next = null;
        }
        return p.next;

    }

    //23. 合并 K 个升序链表(递归)
    public ListNode mergeKLists2(ListNode[] lists) {
        return mergeKLists2(lists, 0, lists.length - 1);

    }

    public ListNode mergeKLists2(ListNode[] lists, int l, int r) {
        if (l < r) {
            int mid = l + (r - l) / 2;
            return mergeTwoLists(mergeKLists2(lists, l, mid), mergeKLists2(lists, mid + 1, r));
        }
        return l == r ? lists[l] : null;
    }


    public static ListNode construct(int[] nums) {
        ListNode head = new ListNode(), p = head;
        for (int i = 0; i < nums.length; i++) {
            p.next = new ListNode(nums[i], null);
            p = p.next;
        }
        return head.next;
    }

    //160. 相交链表(不熟练)
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA, p2 = headB;
        while (p1 != p2) {
            if (p1 == null) {
                p1 = headB;
            } else {
                p1 = p1.next;
            }
            if (p2 == null) {
                p2 = headA;
            } else {
                p2 = p2.next;
            }
        }
        return p1;

    }

    //    141. 环形链表
    public boolean hasCycle(ListNode head) {
        ListNode s = head, f = head;
        while (f != null) {
            s = s.next;
            f = f.next;
            if (f == null) {
                return false;
            }
            f = f.next;
            if (s == f) {
                return true;
            }
        }
        return false;

    }

    //142. 环形链表 II
    public ListNode detectCycle(ListNode head) {
        ListNode s = head, f = head, p = head;
        while (f != null) {
            s = s.next;
            if (f.next != null) {
                f = f.next.next;
            } else {
                return null;
            }

            if (s == f) {
                while (p != s) {
                    s = s.next;
                    p = p.next;
                }
                return p;
            }
        }
        return null;

    }

    //143. 重排链表
    public void reorderList(ListNode head) {
        ListNode s = head, f = head, pre = null;
        while (f != null) {
            pre = s;
            s = s.next;
            f = f.next;
            if (f != null) {
                f = f.next;
            }
        }
        if (pre != null) {
            pre.next = null;
        }
//        disListNode(head);
//        disListNode(reverse(s));
        head = merge(head, reverse(s));
//        disListNode(head);
    }

    public ListNode reverse(ListNode a) {
        ListNode pre = null, p = a;
        while (p != null) {
            ListNode next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        return pre;

    }

    public ListNode merge(ListNode a, ListNode b) {
        ListNode nh = new ListNode(), p = nh;
        while (a != null && b != null) {
            ListNode t = a.next;
            p.next = a;
            a.next = null;
            a = t;
            p = p.next;

            t = b.next;
            p.next = b;
            b.next = null;
            b = t;
            p = p.next;
        }
        if (a != null) {
            p.next = a;
        } else {
            p.next = b;
        }
        return nh.next;


    }

    public static void disListNode(ListNode head) {
        ListNode p = head;
        while (p != null) {
            System.out.print(p.val + " ");
            p = p.next;
        }
        System.out.println();
    }

    //69. x 的平方根
    public int mySqrt(int x) {
        long n = x, l = 0, r = n;
        while (l <= r) {
            long mid = l + (r - l) / 2;
            if (mid * mid == n) {
                return (int) mid;
            } else if (mid * mid > n) {
                r = mid - 1;
            } else if (mid * mid < n) {
                l = mid + 1;
            }
        }
        return (int) l - 1;

    }

//    public Integer priv = 6746874;

    public static void main(String[] args) throws ClassNotFoundException, InstantiationException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        int[] nums1 = new int[]{5, 1, 1, 2, 0, 0};
        ListNode head = construct(nums1);

        List<Integer> list = new ArrayList<>();
//        list.se

        MT o = new MT();
        o.printFile();
//        o.insertSort(nums1);
//        MyUtile.dis(nums1);
//        int a = o.priv;
//        o.partition(head, 2);
//        o.reorderList(head);
//        o.deleteDuplicates(head);

//        o.groupAnagrams();
        Class.forName("java.lang.String").getConstructor(String.class);
        String s = String.class.newInstance();
        s = "1544545";
//        System.out.println(s);
        o.getClass().newInstance();
//        MT obj= (MT) c.newInstance();

//        Class<?> Integer;
//        System.out.println(Arrays.toString());
//        Class c = Class.forName("interview_prepare.MT");

//        System.out.println(c.getMethod("reverse", int.class).invoke(o, 12345));
//        System.out.println(o.longestPalindrome("babad"));
//        Method[] methods = c.getDeclaredMethods();
//        for (int i = 0; i < methods.length; i++) {
//            System.out.println(methods[i]);
//        }
//        System.out.println(s);
    }
}
