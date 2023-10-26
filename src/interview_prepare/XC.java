package interview_prepare;

import category.MyUtile;
import category.Tree;
//import category.TrieNode;
import javafx.util.Pair;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class XC {
    public static void disMap(int[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                System.out.print("(" + i + "," + j + ") " + a[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void swap(int[] nums, int a, int b) {
        int t = nums[a];
        nums[a] = nums[b];
        nums[b] = t;
    }

    //    快排
    public void qSort(int[] nums) {
        randomChange(nums);
        qSort(nums, 0, nums.length - 1);
        dis(nums);

    }

    public void qSort(int[] nums, int l, int r) {
        if (l < r) {
            int p = part(nums, l, r);
            qSort(nums, l, p - 1);
            qSort(nums, p + 1, r);
        }
    }

    public int part(int[] nums, int l, int r) {
        int p = nums[l];
        while (l < r) {
            while (l < r && nums[r] <= p) {
                r--;
            }
            nums[l] = nums[r];
            while (l < r && nums[l] >= p) {
                l++;
            }
            nums[r] = nums[l];
        }
        nums[l] = p;
        return l;
    }

    public void randomChange(int[] nums) {
        Random r = new Random();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            int ind = i + r.nextInt(n - i);
            int t = nums[i];
            nums[i] = nums[ind];
            nums[ind] = t;
        }
        MyUtile.dis(nums);
    }

    //    归并排序
    public int[] sortArray(int[] nums) {
        sortArray(nums, 0, nums.length - 1);
        MyUtile.dis(nums);
        return nums;
    }

    public void sortArray(int[] nums, int l, int r) {
        if (l < r) {
            int mid = l + (r - l) / 2;
            sortArray(nums, l, mid);
            sortArray(nums, mid + 1, r);
            merge(nums, l, mid, r);
        }

    }

    public void merge(int[] nums, int l, int mid, int r) {
        Deque<Integer> q = new LinkedList<>();
        int p1 = l, p2 = mid + 1;
        while (p1 <= mid && p2 <= r) {
            q.offerLast(nums[p1] <= nums[p2] ? nums[p1++] : nums[p2++]);
        }
        while (p1 <= mid) {
            q.offerLast(nums[p1++]);
        }
        while (p2 <= r) {
            q.offerLast(nums[p2++]);
        }
        for (int i = l; i <= r; i++) {
            nums[i] = q.pop();
        }
    }

    //    堆排
    public int[] heapSortArray(int[] nums) {
        int n = nums.length;
        buildHeap(nums);
        for (int i = 0; i < n; i++) {
            MyUtile.swap(nums, 0, n - i - 1);
            down(nums, 0, n - i - 1);
        }
        MyUtile.dis(nums);
        return nums;
    }

    public void buildHeap(int[] nums) {
        int n = nums.length;
        for (int i = n / 2; i >= 0; i--) {
            down(nums, i, n);
        }
    }

    public void down(int[] nums, int par, int n) {
        int c = par * 2 + 1;
        while (c < n) {
            if (c + 1 < n && nums[c + 1] > nums[c]) {
                c++;
            }
            if (nums[c] > nums[par]) {
                MyUtile.swap(nums, c, par);
            }
            par = c;
            c = par * 2 + 1;
        }
    }


    // topK问题1 215. 数组中的第K个最大元素 时间n 空间logn(使用栈的代价是logn)
    public int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        k = n - k;
        return findKthLargest(nums, k, 0, n - 1);
    }

    public int findKthLargest(int[] nums, int k, int l, int r) {
        while (l < r) {
            int p = part(nums, l, r);
            if (p == k) {
                return nums[k];
            } else if (p < k) {
                l = p + 1;
            } else if (p > k) {
                r = p - 1;
            }
        }
        return nums[l];
    }

//    public int findKthLargest(int[] nums, int k, int l, int r) {
//        if (l < r) {
//            int p = part(nums, l, r);
//            if (p == k) {
//                return nums[p];
//            } else if (p > k) {
//                return findKthLargest(nums, k, l, p - 1);
//            } else if (p < k) {
//                return findKthLargest(nums, k, p + 1, r);
//            }
//        }
//        return nums[l];
//    }

    // topK问题2 347. 前 K 个高频元素 最坏n^2 平均n

    public int[] topKFrequent(int[] nums, int k) {
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        List<Pair<Integer, Integer>> list = new ArrayList<>();
        map.forEach((key, v) -> list.add(new Pair<>(v, key)));

        k = map.size() - k;
//        直接转化成求第k大的数 之后直接取后面的所有数 不用在排序的时候保存数据
        findKthLargest(list, k, 0, list.size() - 1);
        System.out.println(list.subList(k, list.size()));
        return list.subList(k, list.size()).stream().mapToInt(Pair::getValue).toArray();
    }

    public void findKthLargest(List<Pair<Integer, Integer>> nums, int k, int l, int r) {
        if (l < r) {
            int p = part(nums, l, r);
            if (p == k) {
                return;
            } else if (p < k) {
                findKthLargest(nums, k, p + 1, r);
            } else if (p > k) {
//                大于k的数后面的不用再part 直接存   实际上不用在这里面存 直接转化成求第k大的数 之后直接取后面的所有数
//                res.addAll(nums.subList(p, nums.size()).stream().map(Pair::getValue).collect(Collectors.toList()));
                findKthLargest(nums, k, l, p - 1);
            }
        }
    }

    public int part(List<Pair<Integer, Integer>> nums, int l, int r) {
        Pair<Integer, Integer> p = nums.get(l);
        while (l < r) {
            while (l < r && nums.get(r).getKey() >= p.getKey()) {
                r--;
            }
//            nums[l] = nums[r];
            nums.set(l, nums.get(r));
            while (l < r && nums.get(l).getKey() <= p.getKey()) {
                l++;
            }
//            nums[r] = nums[l];
            nums.set(r, nums.get(l));
        }
        nums.set(l, p);
        return l;
    }

    // topK问题2 347. 前 K 个高频元素 桶排序做法 时间n 空间n
    public int[] topKFrequentButton(int[] nums, int k) {
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        List<Integer>[] list = new List[n + 1];
        map.forEach((key, v) -> {
            if (list[v] == null) {
                list[v] = new ArrayList<>();
            }
            list[v].add(key);
        });
        System.out.println(map);
//        MyUtile.dis(list);
        List<Integer> res = new ArrayList<>();
        for (int i = n; i >= 0 && res.size() < k; i--) {
            if (list[i] == null) {
                continue;
            }
            res.addAll(list[i]);
        }
        System.out.println(res);
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

    public static void dis(int[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i] + " ");
        }
        System.out.println();
    }

    //    顺子问题 659. 分割数组为连续子序列
    public boolean isPossible(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<>();
        Map<Integer, Integer> need = new HashMap<>();

        int n = nums.length;
        for (int i = 0; i < n; i++) {
            freq.put(nums[i], freq.getOrDefault(nums[i], 0) + 1);
        }
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (freq.get(num) == 0) {
                continue;
            }
            if (need.containsKey(num) && need.get(num) > 0) {
                freq.compute(num, (k, v) -> v - 1);
                need.computeIfPresent(num, (k, v) -> v - 1);
                need.put(num + 1, need.getOrDefault(num + 1, 0) + 1);
            } else if (freq.containsKey(num + 1) && freq.get(num + 1) > 0 && freq.containsKey(num + 2) && freq.get(num + 2) > 0) {
                freq.compute(num, (k, v) -> v - 1);
                freq.compute(num + 1, (k, v) -> v - 1);
                freq.compute(num + 2, (k, v) -> v - 1);
                need.put(num + 3, need.getOrDefault(num + 3, 0) + 1);
            } else {
                return false;
            }
        }
        return true;
    }

    //    环检测 207. 课程表
    boolean[] v, path;
    boolean cycle = false;

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        v = new boolean[numCourses];
        path = new boolean[numCourses];
        List<Integer>[] g = buildGraph(numCourses, prerequisites);
        for (int i = 0; i < numCourses; i++) {
            hasCycle(g, i);
        }
        return !cycle;
    }

    public void hasCycle(List<Integer>[] g, int val) {
        if (path[val]) {
            cycle = true;
        }
        if (v[val] || cycle) {
            return;
        }
        v[val] = true;
        path[val] = true;
        for (Integer to : g[val]) {
            hasCycle(g, to);
        }
        path[val] = false;
    }

    public List<Integer>[] buildGraph(int numCourses, int[][] prerequisites) {
        List<Integer>[] g = new ArrayList[numCourses];
        for (int i = 0; i < numCourses; i++) {
            g[i] = new ArrayList<>();
        }
        for (int[] prerequisite : prerequisites) {
//            if (g[prerequisite[1]] == null) {
//                g[prerequisite[1]] = new ArrayList<>();
//            }
            g[prerequisite[1]].add(prerequisite[0]);
        }
        return g;
    }

    //    210. 课程表 II 拓扑排序
    boolean[] topv, topPa;
    boolean isCycle = false;
    List<Integer> postOrder;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        topv = new boolean[numCourses];
        topPa = new boolean[numCourses];
        postOrder = new ArrayList<>();
        List<Integer>[] g = buildG(numCourses, prerequisites);
        for (int i = 0; i < numCourses; i++) {
            topSort(g, i);
            if (isCycle) {
                return new int[0];
            }
        }

        Collections.reverse(postOrder);
        System.out.println(postOrder);
        return postOrder.stream().mapToInt(Integer::intValue).toArray();
    }

    public void topSort(List<Integer>[] g, int from) {
        if (topPa[from]) {
            isCycle = true;
        }
        if (topv[from] || isCycle) {
            return;
        }
        topv[from] = true;
        topPa[from] = true;
        for (Integer to : g[from]) {
            topSort(g, to);
        }
        postOrder.add(from);
        topPa[from] = false;
    }

    public List<Integer>[] buildG(int numCourses, int[][] prerequisites) {
        List<Integer>[] g = new ArrayList[numCourses];
        for (int i = 0; i < numCourses; i++) {
            g[i] = new ArrayList<>();
        }
        for (int[] p : prerequisites) {
            g[p[1]].add(p[0]);
        }
        return g;
    }


    //    141. 环形链表
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null) {
            if (fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;

    }

    //    kmp 28. 找出字符串中第一个匹配项的下标
    public int strStr(String haystack, String needle) {
        int[] next = getnext(needle);
        int j = 0;
        for (int i = 0; i < haystack.length(); i++) {
            while (j > 0 && haystack.charAt(i) != needle.charAt(j)) {
                j = next[j - 1];
            }
            if (haystack.charAt(i) == needle.charAt(j)) {
                j++;
            }
            if (j >= next.length) {
                return i - j;
            }
        }
        return -1;

    }

    public int[] getnext(String s) {
        int n = s.length();
        int[] next = new int[n];
        next[0] = 0;
        int preNext = next[0];
        for (int i = 1; i < n; i++) {
            while (preNext > 0 && s.charAt(i) != s.charAt(preNext)) {
                preNext = next[preNext - 1];
            }
            if (s.charAt(i) == s.charAt(preNext)) {
                preNext++;
            }
            next[i] = preNext;
        }
        dis(next);

        return next;
    }

    //33. 搜索旋转排序数组 二分
    public int search(int[] nums, int target) {
        int n = nums.length, l = 0, r = n - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] >= nums[0]) {
//                分左右两侧讨论
                if (target > nums[mid] || target < nums[0]) {
//                    右侧
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            } else {
                if (target < nums[mid] || target > nums[n - 1]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            }
        }
        return l >= 0 && l < n && nums[l] == target ? l : -1;
    }

    //153. 寻找旋转排序数组中的最小值(最大值直接转换成最小值之后取前一位)
//    总结：找最大（小）值就与大（左）的分支，也就是l（r）做比较
    public int findMin(int[] nums) {
        int n = nums.length, l = 0, r = n - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] > nums[r]) {
                l = mid + 1;
            } else if (nums[mid] < nums[r]) {
                r = mid;
            } else if (nums[mid] == nums[r]) {
                r--;
            }
        }
        return nums[l];

    }

    //300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        int n = nums.length, res = 0;
        int dp[] = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] = nums[j] < nums[i] && dp[j] + 1 > dp[i] ? dp[j] + 1 : dp[i];
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //    53. 最大子数组和
    public int maxSubArray(int[] nums) {
        int n = nums.length, res = nums[0];
        int[] dp = new int[n];
        dp[0] = nums[0];
        for (int i = 1; i < n; i++) {
            dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
            res = Math.max(dp[i], res);
        }
        return res;

    }

    //64. 最小路径和
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0) {
                    dp[i][j] = j == 0 ? grid[0][0] : grid[0][j] + dp[0][j - 1];
                    continue;
                }
                if (j == 0) {
                    dp[i][j] = grid[i][0] + dp[i - 1][0];
                    continue;
                }
                dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j];
            }
        }
        disMap(dp);
        return dp[m - 1][n - 1];

    }

    //72.编辑距离
    public int minDistance(String word1, String word2) {
        return minDistance(word1, 0, word2, 0, new int[word1.length()][word2.length()]);
    }

    public int minDistance(String word1, int i, String word2, int j, int[][] memo) {
        int m = word1.length(), n = word2.length();
        if (i == m) {
            return n - j;
        }
        if (j == n) {
            return m - i;
        }
        if (memo[i][j] != 0) {
            return memo[i][j];
        }
        if (word1.charAt(i) == word2.charAt(j)) {
            return memo[i][j] = minDistance(word1, i + 1, word2, j + 1, memo);
        }
        return memo[i][j] = Math.min(Math.min(minDistance(word1, i, word2, j + 1, memo), minDistance(word1, i + 1, word2, j, memo)), minDistance(word1, i + 1, word2, j + 1, memo)) + 1;
    }
    //72.编辑距离 dp做法

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
                if (word1.charAt(i) == word2.charAt(j)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];
    }

    //    5. 最长回文子串
    public String longestPalindrome(String s) {
        int n = s.length(), res = 0, l = 0, r = 0;
        boolean[][] dp = new boolean[n][n];

        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        for (int i = 1; i < n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = i - j == 1 || dp[i - 1][j + 1];
                }
                if (dp[i][j] && i - j > res) {
                    res = i - j;
                    l = j;
                    r = i;
                }
            }
        }
//        MyUtile.disMap(dp);
//        System.out.println(l + ":" + r);
        return s.substring(l, r + 1);
    }

    //219. 存在重复元素 II 滑动窗口
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        int l = 0, r = 0, n = nums.length;
        Map<Integer, Integer> win = new HashMap<>();
        while (r < n) {
            int rnum = nums[r];
            if (r - l > k) {
                win.put(nums[l], win.get(nums[l]) - 1);
                l++;
            }
            if (win.containsKey(rnum) && win.get(rnum) > 0) {
                return true;
            } else {
                win.put(rnum, 1);
            }
            r++;
        }
        return false;

    }

    //8. 字符串转换整数 (atoi)
    public int myAtoi(String s) {
        s = s.trim();
        if (s.length() == 0) {
            return 0;
        }

        long res = 0;
        int i = 0, n = s.length();
        int sign = 1;
        if (s.charAt(i) == '-') {
            sign = -1;
            i++;
        } else if (s.charAt(i) == '+') {
            i++;
        }
        while (i < n && s.charAt(i) >= '0' && s.charAt(i) <= '9') {
            res = res * 10 + s.charAt(i) - '0';
            if (res > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            i++;
        }
        return (int) res * sign;

    }

    //108. 将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    public TreeNode sortedArrayToBST(int[] nums, int l, int r) {
        if (l <= r) {
            int mid = l + (r - l) / 2;
            return new TreeNode(nums[mid], sortedArrayToBST(nums, l, mid - 1), sortedArrayToBST(nums, mid + 1, r));
        } else return null;
    }

    //14. 最长公共前缀
    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 1) {
            return strs[0];
        }
        int l = 0, r = Arrays.stream(strs).mapToInt(String::length).min().getAsInt() - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (longestCommonPrefix(strs, l, mid)) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return strs[0].substring(0, l);
    }

    public boolean longestCommonPrefix(String[] strs, int start, int end) {
        String com = strs[0].substring(start, end + 1);
        for (int i = 0; i < strs.length; i++) {
            if (!com.equals(strs[i].substring(start, end + 1))) {
                return false;
            }
        }
        return true;
    }

    //    46. 全排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        permute(nums, 0, new boolean[nums.length], new LinkedList<>(), res);
        System.out.println(res);
        return res;
    }

    public void permute(int[] nums, int i, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        if (i == n) {
//            res.add(new ArrayList<>(q));
            res.add(Arrays.stream(nums).boxed().collect(Collectors.toList()));
            return;
        }
        for (int j = i; j < n; j++) {
//            if (!v[j]) {
//                v[j] = true;
            swap(nums, i, j);
//            q.offerLast(nums[j]);
            permute(nums, i + 1, v, q, res);
//            q.pollLast();
            swap(nums, i, j);
//                v[j] = false;
//            }
        }
    }

    //    47. 全排列 II
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        permuteUnique(nums, 0, new boolean[nums.length], new LinkedList<>(), res);
        System.out.println(res);
        return res;
    }

    public void permuteUnique(int[] nums, int i, boolean[] v, Deque<Integer> q, List<List<Integer>> res) {
        int n = nums.length;
        if (q.size() == n) {
            res.add(new ArrayList<>(q));
            return;
        }
        int pre = Integer.MAX_VALUE;
        for (int j = 0; j < n; j++) {
            if (!v[j] && pre != nums[j]) {
                v[j] = true;
                q.offerLast(pre = nums[j]);
                permuteUnique(nums, i + 1, v, q, res);
                q.pollLast();
                v[j] = false;
            }
        }
    }

    //77. 组合
    public List<List<Integer>> combine(int n, int k) {
        int[] nums = IntStream.range(1, n + 1).toArray();
        System.out.println(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        List<List<Integer>> res = new ArrayList<>();
        combine(nums, 0, k, new LinkedList<>(), res);
        System.out.println(res);
        return res;
    }

    public void combine(int[] nums, int now, int k, Deque<Integer> q, List<List<Integer>> res) {
        if (q.size() == k) {
            res.add(new ArrayList<>(q));
            return;
        }
        for (int i = now; i < nums.length; i++) {
            q.offerLast(nums[i]);
            combine(nums, i + 1, k, q, res);
            q.pollLast();
        }
    }
//39. 组合总和

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
//        int[] nums = IntStream.range(1, n + 1).toArray();
//        System.out.println(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        List<List<Integer>> res = new ArrayList<>();
        combinationSum(candidates, 0, target, new LinkedList<>(), res);
        System.out.println(res);
        return res;
    }

    public void combinationSum(int[] nums, int now, int tar, Deque<Integer> q, List<List<Integer>> res) {
        int sum = q.stream().mapToInt(Integer::intValue).sum();
        if (sum >= tar) {
            if (sum == tar) {
                res.add(new ArrayList<>(q));
            }
            return;
        }

        int pre = Integer.MAX_VALUE;
        for (int i = now; i < nums.length; i++) {
            if (pre == nums[i]) {
                continue;
            }
            q.offerLast(pre = nums[i]);
            combinationSum(nums, i, tar, q, res);
            q.pollLast();
        }
    }

    //216. 组合总和 III
    public List<List<Integer>> combinationSum3(int k, int n) {
        int[] nums = IntStream.range(1, 10).toArray();
        List<List<Integer>> res = new ArrayList<>();
        combinationSum3(nums, 0, n, k, new LinkedList<>(), res);
        System.out.println(res);
        return res;
    }

    public void combinationSum3(int[] nums, int now, int tar, int len, Deque<Integer> q, List<List<Integer>> res) {
        int sum = q.stream().mapToInt(Integer::intValue).sum();
        if (sum >= tar) {
            if (len == q.size() && sum == tar) {
                res.add(new ArrayList<>(q));
            }
            return;
        }

        for (int i = now; i < nums.length; i++) {
            q.offerLast(nums[i]);
            combinationSum3(nums, i + 1, tar, len, q, res);
            q.pollLast();
        }
    }


    public int strStr2(String haystack, String needle) {
        int[] next = getNext2(needle);
        int cur = 0;
        for (int i = 0; i < haystack.length(); i++) {
            while (cur > 0 && haystack.charAt(i) != needle.charAt(cur)) {
                cur = next[cur - 1];
            }
            if (haystack.charAt(i) == needle.charAt(cur)) {
                cur++;
            }
            if (cur >= needle.length()) {
                return i - cur;
            }
        }
        return -1;
    }

    public int[] getNext2(String p) {
        int n = p.length();
        int[] next = new int[n];
        int pre = next[0];
        for (int i = 1; i < n; i++) {
            while (pre > 0 && p.charAt(i) != p.charAt(pre)) {
                pre = next[pre - 1];
            }
            if (p.charAt(i) == p.charAt(pre)) {
                pre++;
            }
            next[i] = pre;
        }
        return next;
    }

    public static void main(String[] args) {
        XC xc = new XC();
        int[] nums = new int[]{7, 5, -10, 9, 0, 1, -45, 89, 2};
        int[] nums1 = new int[]{1, 1, 2};
//        int[] nums1 = new int[]{4,4,7,4,4,4,4};
//        MyUtile.dis(nums);
//        xc.qSort(nums);
//        System.out.println(xc.findMin(nums1));
//        MyUtile.dis(nums);
//        xc.topKFrequent(nums1, 1);
//        xc.topKFrequentButton(nums1, 2);
//        xc.sortArray(nums);
//        xc.heapSortArray(nums);

        xc.strStr2("mississippi", "issip");
//        xc.permuteUnique(nums1);
//        xc.longestCommonPrefix(new String[]{"ab", "a"});

        int[][] prerequisites = new int[][]{{1, 0}, {0, 1}};
//        xc.findOrder(2, prerequisites);

//        System.out.println(xc.containsNearbyDuplicate(nums1, 3));

    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

//LCR 062. 实现 Trie (前缀树)
class TrieNode {
    //    前缀树：实际是一颗多叉树，树枝为一个字符，节点值为当前这个路径删是否有单词，使用end表示
    TrieNode[] children = new TrieNode[26];
    boolean end;

    public TrieNode() {
    }

    public TrieNode(TrieNode[] children, boolean end) {
        this.children = children;
        this.end = end;
    }
}

class Trie {
    TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            p = (p.children[word.charAt(i) - 'a'] = p.children[word.charAt(i) - 'a'] == null ? new TrieNode() : p.children[word.charAt(i) - 'a']);
        }
        p.end = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            if (p == null || (p = (p.children[word.charAt(i) - 'a'])) == null) {
                return false;
            }
        }
        return p.end;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode p = root;
        for (int i = 0; i < prefix.length(); i++) {
            if ((p = (p.children[prefix.charAt(i) - 'a'])) == null) {
                return false;
            }
        }
        return true;
    }

}

//211. 添加与搜索单词 - 数据结构设计
class WordDictionary {
    TrieNode root;

    public WordDictionary() {
        root = new TrieNode();
    }

    public void addWord(String word) {
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            p = (p.children[word.charAt(i) - 'a'] = p.children[word.charAt(i) - 'a'] == null ? new TrieNode() : p.children[word.charAt(i) - 'a']);
        }
        p.end = true;
    }

    public boolean search(String word) {
        TrieNode p = root;
        return search(word, p);
    }

    public boolean search(String word, TrieNode root) {
        TrieNode p = root;
        if (p == null) {
            return false;
        }
        for (int i = 0; i < word.length(); i++) {
            if (p == null) {
                return false;
            }
            if (word.charAt(i) == '.') {
                int finalI = i;
                return Arrays.stream(p.children).anyMatch(cj -> cj != null && search(word.substring(finalI + 1), cj));
            } else {
                p = (p.children[word.charAt(i) - 'a']);
            }
        }
        return p != null && p.end;
    }


}

class ListNode {
    int val;
    ListNode next;

    public ListNode() {
    }

    public ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

class CQueue {

    Stack<Integer> s1, s2;

    public CQueue() {
        s1 = new Stack<>();
        s2 = new Stack<>();
    }

    public void appendTail(int value) {
        s1.push(value);
    }

    public int deleteHead() {
        if (s1.size() == 0) {
            return -1;
        }
        while (s1.size() > 1) {
            s2.push(s1.pop());
        }
        int res = s1.pop();
        while (!s2.isEmpty()) {
            s1.push(s2.pop());
        }
        return res;
    }
}


