package category;


import java.util.*;

// 单链表节点的结构
class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

public class LinkLists {
    //    链表相关：
//    86. 分隔链表
    public ListNode partition(ListNode head, int x) {
//    拆成两链表
        ListNode l = new ListNode(-1);
        ListNode g = new ListNode(-1);
        ListNode p = head, lp = l, gp = g;
        while (p != null) {
            if (p.val < x) {
                lp.next = p;
                lp = lp.next;
            } else {
                gp.next = p;
                gp = gp.next;
            }
            ListNode t = p.next;//顺序问题需注意
            p.next = null;
            p = t;
        }
        lp.next = g.next;
        return l.next;
    }

    //16. 最接近的三数之和
    public int threeSumClosest(int[] nums, int target) {

//    双指针解决
        Arrays.sort(nums);
        int best = 1000000000;
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int tempt = target - nums[i];
            int l = i + 1, r = nums.length - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum == target) return target;
                best = Math.abs(sum - target) < Math.abs(best - target) ? sum : best;
                if (sum < target) {
                    l++;
                } else {
                    r--;
                }
            }
        }
        return best;
    }

    //27. 移除元素
    public int removeElement(int[] nums, int val) {
        if (nums.length == 0) return 0;
        int l = 0, r = nums.length - 1;
        if (l == r && nums[l] != val) return nums.length;//边界条件
        while (l < r) {
            while (l < r && nums[l] != val) l++;
            while (l < r && nums[r] == val) r--;
            int t = nums[l];
            nums[l] = nums[r];
            nums[r] = t;
        }
        return l == nums.length - 1 && nums[l] != val ? l + 1 : l;//边界条件
    }

    //206. 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode pre = null, p = head;
        while (Objects.nonNull(p)) {
            ListNode next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        return pre;
    }

    //92. Reverse Linked List II
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummyNode = new ListNode();
        dummyNode.next = head;
        ListNode pre = dummyNode;
        for (int i = 0; i < left - 1; i++) {
            pre = pre.next;
        }
        ListNode p = pre.next;//pre固定 区间内做头插逆序
        for (int i = left; i < right && Objects.nonNull(p); i++) {
            ListNode next = p.next;
            p.next = next.next;
            next.next = pre.next;
            pre.next = next;
        }
        return dummyNode.next;
    }

    public ListNode reverse(ListNode a, ListNode b) {
//        反转[a,b)
        ListNode pre = null, c = a, next = a;
        while (!Objects.equals(b, c)) {
            next = c.next;
            c.next = pre;
            pre = c;
            c = next;
        }
        return pre;
    }

    //25. Reverse Nodes in k-Group k个一组反转链表
    public ListNode reverseKGroup(ListNode head, int k) {
        if (Objects.isNull(head)) return null;
        ListNode b = head, a = head;
        for (int i = 0; i < k; i++) {
            if (Objects.isNull(b)) {
                return head;
            }
            b = b.next;
        }
        ListNode newHead = reverse(a, b);
        a.next = reverseKGroup(b, k);//a变成了尾巴 因此要指向下一个头
        return newHead;
    }

    public void dis(ListNode head) {
        ListNode p = head;
        while (Objects.nonNull(p)) {
            System.out.println(p.val);
            p = p.next;
        }
    }

    //    1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i], i);
        }

        return new int[]{};
    }

    //    15. 三数之和
    public List<List<Integer>> threeSum(int[] nums) {

        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], i);
        }

        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        Set<List<Integer>> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            int target = 0 - nums[i];
            int l = i + 1, r = nums.length - 1;
            while (l < r) {
                if (nums[l] + nums[r] == target) {

                    List<Integer> tmplist = new ArrayList<>();
                    tmplist.add(nums[i]);
                    tmplist.add(nums[l]);
                    tmplist.add(nums[r]);
                    set.add(tmplist);
                    l++;
                } else if (nums[l] + nums[r] < target) {
                    l++;
                } else {
                    r--;
                }
            }
        }
        res.addAll(set);
        return res;
    }

    //167. 两数之和 II - 输入有序数组
    public int[] twoSum2(int[] numbers, int target) {

        int l = 0, r = numbers.length - 1;
        while (l < r) {
            if (numbers[l] + numbers[r] == target) {
                return new int[]{l + 1, r + 1};
            } else if (numbers[l] + numbers[r] < target) {
                l++;
            } else {
                r--;
            }
        }
        return new int[]{};
    }

    //5. 最长回文子串 动态规划
//    https://leetcode.cn/problems/longest-palindromic-substring/solution/zhong-xin-kuo-san-fa-he-dong-tai-gui-hua-by-reedfa/
    public String longestPalindrome(String s) {
        System.out.println(s);
        int start = 0, end = 0, maxlen = 1, strlen = s.length();
        if (s.length() < 2) return s.substring(start, end+1);

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

    public static void main(String[] args) {

        List<Integer> integers = new ArrayList<>();

        int[] ids = new int[]{2, 7, 11, 15};
        ListNode dummyNode = new ListNode(-1);
        ListNode p = dummyNode;
        for (int e : ids) {
            p.next = new ListNode(e);
            p = p.next;
        }
        LinkLists obj = new LinkLists();
        System.out.println();
//        obj.dis(obj.reverseKGroup(dummyNode.next, 2));
        System.out.println(obj.longestPalindrome("babad"));
    }


}

