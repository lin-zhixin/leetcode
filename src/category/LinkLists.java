package category;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

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

    public static void main(String[] args) {

        List<Integer> integers = new ArrayList<>();

        int[] ids = new int[]{1, 2, 3, 4, 5};
        ListNode dummyNode = new ListNode(-1);
        ListNode p = dummyNode;
        for (int e : ids) {
            p.next = new ListNode(e);
            p = p.next;
        }
        LinkLists obj = new LinkLists();
        System.out.println();
        obj.dis(obj.reverseKGroup(dummyNode.next, 2));
    }


}

