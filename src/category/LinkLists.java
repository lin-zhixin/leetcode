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

    public ListNode detectCycle(ListNode head) {
        ListNode fast = head, slow = head;
        while (fast != null) {
            slow = slow.next;
            if (fast.next == null || fast.next.next == null) {
                return null;
            }
            fast = fast.next.next;
//            数学相关 具体看官方解析
            if (fast == slow) {
                ListNode t = head;
                while (t != slow) {
                    slow = slow.next;
                    t = t.next;
                }
                return t;
            }
//            System.out.println(memo);
        }
        return null;
    }

    //    2023.9.3之前：
    //    //    链表相关：
////    86. 分隔链表
//    public ListNode partition(ListNode head, int x) {
////    拆成两链表
//        ListNode l = new ListNode(-1);
//        ListNode g = new ListNode(-1);
//        ListNode p = head, lp = l, gp = g;
//        while (p != null) {
//            if (p.val < x) {
//                lp.next = p;
//                lp = lp.next;
//            } else {
//                gp.next = p;
//                gp = gp.next;
//            }
//            ListNode t = p.next;//顺序问题需注意
//            p.next = null;
//            p = t;
//        }
//        lp.next = g.next;
//        return l.next;
//    }
//
//    //16. 最接近的三数之和
//    public int threeSumClosest(int[] nums, int target) {
//
////    双指针解决
//        Arrays.sort(nums);
//        int best = 1000000000;
//        for (int i = 0; i < nums.length; i++) {
//            if (i > 0 && nums[i] == nums[i - 1]) {
//                continue;
//            }
//            int tempt = target - nums[i];
//            int l = i + 1, r = nums.length - 1;
//            while (l < r) {
//                int sum = nums[i] + nums[l] + nums[r];
//                if (sum == target) return target;
//                best = Math.abs(sum - target) < Math.abs(best - target) ? sum : best;
//                if (sum < target) {
//                    l++;
//                } else {
//                    r--;
//                }
//            }
//        }
//        return best;
//    }
//
//    //27. 移除元素
//    public int removeElement(int[] nums, int val) {
//        if (nums.length == 0) return 0;
//        int l = 0, r = nums.length - 1;
//        if (l == r && nums[l] != val) return nums.length;//边界条件
//        while (l < r) {
//            while (l < r && nums[l] != val) l++;
//            while (l < r && nums[r] == val) r--;
//            int t = nums[l];
//            nums[l] = nums[r];
//            nums[r] = t;
//        }
//        return l == nums.length - 1 && nums[l] != val ? l + 1 : l;//边界条件
//    }
//
//    //206. 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode pre = null, next = null, p = head;
        while (p != null) {
            next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        dis(pre);
        return pre;
    }

    //
//    //234. 回文链表
//    public boolean isPalindrome(ListNode head) {
//        ListNode p1pre = null, prep2 = null, p1 = head, p2 = head;
//        while (Objects.nonNull(p1) && Objects.nonNull(p2)) {
//            p1pre = p1;
////            prep2 = p2;
//            p1 = p1.next;
//            p2 = p2.next;
//            if (Objects.nonNull(p2)) p2 = p2.next;
////            if (Objects.isNull(p2))break;
//        }
//        dis(head);
//
//        reverseLinkList(p1pre, p1, p2, Objects.nonNull(p2) ? p2.next : null);
//        dis(head);
//
//        p1 = p1pre.next;
//        p2 = head;
//        while (Objects.nonNull(p1) && Objects.equals(p1.val, p2.val)) {
//            p1 = p1.next;
//            p2 = p2.next;
//        }
//        return Objects.isNull(p1);
//    }
//
//    public void reverseLinkList(ListNode startpre, ListNode start, ListNode end, ListNode endnext) {
//        ListNode pre = endnext, p = start, next;
//        while (!Objects.equals(p, endnext)) {
//            next = p.next;
//            p.next = pre;
//            pre = p;
//            p = next;
//        }
//        startpre.next = pre;
//    }
//
//    //92. Reverse Linked List II
//    public ListNode reverseBetween(ListNode head, int left, int right) {
//        ListNode dummyNode = new ListNode();
//        dummyNode.next = head;
//        ListNode pre = dummyNode;
//        for (int i = 0; i < left - 1; i++) {
//            pre = pre.next;
//        }
//        ListNode p = pre.next;//pre固定 区间内做头插逆序
//        for (int i = left; i < right && Objects.nonNull(p); i++) {
//            ListNode next = p.next;
//            p.next = next.next;
//            next.next = pre.next;
//            pre.next = next;
//        }
//        return dummyNode.next;
//    }
//
    public ListNode reverse(ListNode a, ListNode b) {
//        反转[a,b)
        ListNode pre = b, next = null, p = a;
        while (p != null) {
            next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        return null;
    }

    //
//    //25. Reverse Nodes in k-Group k个一组反转链表
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode slow = head, fast = head;
        while (fast != null) {
            for (int i = 0; i < k && fast != null; i++) {
                fast = fast.next;
            }
            reverse(slow, fast);
            slow = fast;

        }
        dis(head);
        return head;
    }


    //
    public void dis(ListNode head) {
        ListNode p = head;
        while (Objects.nonNull(p)) {
            System.out.print(p.val + " ");
            p = p.next;
        }
        System.out.println();
    }


    //11. 盛最多水的容器 双指针
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1, max = 0;
        while (l < r) {
            max = Math.max(max, Math.min(height[l], height[r]) * (r - l));
            if (height[l] <= height[r]) {
                l++;
            } else {
                r--;
            }
        }
//        System.out.println(max);
        return max;

    }

    //19. 删除链表的倒数第 N 个结点 (双指针间隔为n+1)
    public ListNode removeNthFromEnd(ListNode head, int n) {
//        tempHead是为了暂时标记这个方法内的head 因为java只有值传递，因此每个方法会为入参复制一份，方法内的head 这边一会儿返回的是方法内的head
        ListNode l, r = head, tempHead = new ListNode(0, head);
        l = tempHead;
        while (Objects.nonNull(r)) {
            if (n > 0) {
                r = r.next;
                n--;
            } else {
                l = l.next;
                r = r.next;
            }
        }
        l.next = l.next.next;
//        l.next = Objects.nonNull(l.next) ? l.next.next : null;
//        System.out.println("方法：");
//        head=tempHead.next;
//        dis(head);
        return tempHead.next;
    }

    //160. 相交链表 总共走的长度相等说明相交
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode pa = headA, pb = headB;
        while (!Objects.equals(pa, pb)) {
            if (pa == null) pa = headB;
            else {
                pa = pa.next;
            }
            if (pb == null) pb = headA;
            else {
                pb = pb.next;
            }
        }
        return pa;
    }

    //2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode res = new ListNode();
        ListNode p1 = l1, p2 = l2, r = res;
        int carry = 0;
        while (p1 != null && p2 != null) {
            int tmp = p1.val + p2.val + carry;
            carry = tmp / 10;
            r.next = new ListNode(tmp % 10);
            r = r.next;
            p1 = p1.next;
            p2 = p2.next;
        }
        while (p1 != null) {
            int tmp = p1.val + carry;
            carry = tmp / 10;
            r.next = new ListNode(tmp % 10);
            r = r.next;
            p1 = p1.next;
        }
        while (p2 != null) {
            int tmp = p2.val + carry;
            carry = tmp / 10;
            r.next = new ListNode(tmp % 10);
            r = r.next;
            p2 = p2.next;
        }
        if (carry != 0) {
            r.next = new ListNode(carry);
            r = r.next;
        }
//        dis(res);
        return res.next;
    }

    //237. 删除链表中的节点
    public void deleteNode(ListNode node) {
        node.val = node.val ^ node.next.val;
        node.next.val = node.val ^ node.next.val;
        node.val = node.val ^ node.next.val;
        node.next = node.next.next;

    }

    //23. 合并 K 个升序链表
    public ListNode merge(ListNode a, ListNode b) {
        ListNode p1 = a, p2 = b, res = new ListNode();
        ListNode p3 = res;
        while (Objects.nonNull(p1) && Objects.nonNull(p2)) {
            if (p1.val <= p2.val) {
                p3.next = new ListNode(p1.val);
                p3 = p3.next;
                p1 = p1.next;
            } else {
                p3.next = new ListNode(p2.val);
                p3 = p3.next;
                p2 = p2.next;
            }
        }
        while (Objects.nonNull(p1)) {
            p3.next = new ListNode(p1.val);
            p3 = p3.next;
            p1 = p1.next;
        }
        while (Objects.nonNull(p2)) {
            p3.next = new ListNode(p2.val);
            p3 = p3.next;
            p2 = p2.next;
        }
        return res.next;
    }

    public ListNode mergeSortLinkLists(ListNode[] lists, int l, int r, int level) {
//        MyUtile.printBlank(level++, "in l=" + l + ",r=" + r);

        if (l > r) {
//            MyUtile.printReturn(level - 1, null);
            return null;
        }
        if (l == r) {
//            MyUtile.printReturn(level - 1, "lists[l]");
            return lists[l];
        }
        int mid = (l + r) / 2;
        ListNode left = mergeSortLinkLists(lists, l, mid, level);
        ListNode right = mergeSortLinkLists(lists, mid + 1, r, level);
//        MyUtile.printReturn(level-1, "merge");
//        MyUtile.printBlank(--level, "out l=" + l + ",r=" + r);
        return merge(left, right);

    }

    public ListNode mergeKLists(ListNode[] lists) {
        return mergeSortLinkLists(lists, 0, lists.length - 1, 0);
    }

    //    堆做法
    public ListNode mergeKLists2(ListNode[] lists) {
//        Queue<ListNode> heap=new PriorityQueue<>(new Comparator<ListNode>() {
//            @Override
//            public int compare(ListNode o1, ListNode o2) {
//                return o1.val-o2.val;
//            }
//        });
        Queue<ListNode> heap = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
        for (int i = 0; i < lists.length; i++) {
            if (Objects.nonNull(lists[i])) {
                heap.offer(lists[i]);
            }
        }
        ListNode res = new ListNode();
        ListNode p = res;
        while (!heap.isEmpty()) {
            ListNode tmp = heap.poll();
            p.next = new ListNode(tmp.val);
            p = p.next;
            if (Objects.nonNull(tmp.next)) {
                heap.offer(tmp.next);
            }
        }
        return res.next;
    }

    public ListNode merge2(ListNode head1, ListNode head2) {
        ListNode res = new ListNode(), p1 = head1, p2 = head2;
        ListNode tempres = res;
        while (Objects.nonNull(p1) && Objects.nonNull(p2)) {
            if (p1.val <= p2.val) {
                tempres.next = new ListNode(p1.val);
                p1 = p1.next;
            } else {
                tempres.next = new ListNode(p2.val);
                p2 = p2.next;
            }
            tempres = tempres.next;
        }
        while (Objects.nonNull(p1)) {
            tempres.next = new ListNode(p1.val);
            p1 = p1.next;
            tempres = tempres.next;
        }
        while (Objects.nonNull(p2)) {
            tempres.next = new ListNode(p2.val);
            p2 = p2.next;
            tempres = tempres.next;
        }
        return res.next;
    }

    public ListNode sortList(ListNode head, ListNode tail, int c) {
//        MyUtile.printBlank(c++, head.val + " " + (Objects.nonNull(tail) ? tail.val : null));
        if (Objects.isNull(head)) {
//            MyUtile.printReturn(c - 1, "Objects.isNull(head)");
            return head;
        }
        if (Objects.equals(head, tail)) {
//            MyUtile.printReturn(c - 1, "Objects.equals(head, tail)");
            return head;
        }
//        坑；只有两个数的时候要拆成一边一个 所以给head置为null之后就是head一个排序 剩下的tail别的递归会排的不用关注
        if (head.next == tail) {
//            MyUtile.printReturn(c - 1, "head.next == tail");

            head.next = null;
            return head;
        }

        ListNode slow = head, fast = head;
        while (!Objects.equals(fast, tail)) {
            slow = slow.next;
            fast = fast.next;
            if (!Objects.equals(fast, tail)) {
                fast = fast.next;
            }
//            if (Objects.isNull(fast))break;
        }

        ListNode left = sortList(head, slow, c);
        ListNode right = sortList(slow, tail, c);
//        MyUtile.printBlank(--c, head.val + " " + (Objects.nonNull(tail) ? tail.val : null));
//        MyUtile.printReturn(c, "merge");
        return merge2(left, right);
    }

    public ListNode sortList(ListNode head) {
        return sortList(head, null, 0);
    }

    public static void main(String[] args) {

        List<Integer> integers = new ArrayList<>();

//        int[] ids = new int[]{1,2,3,4,5};
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
        for (int e : ids) {
            p2.next = new ListNode(e);
            p2 = p2.next;
        }
        int[] ids3 = new int[]{2, 6};
        ListNode dummyNode3 = new ListNode(-1);
        ListNode p3 = dummyNode3;
        for (int e : ids) {
            p3.next = new ListNode(e);
            p3 = p3.next;
        }

        LinkLists obj = new LinkLists();
        System.out.println();
        obj.dis(obj.reverseKGroup(dummyNode.next, 2));
//        System.out.println(obj.maxArea(new int[]{1, 1}));
//        obj.dis(obj.removeNthFromEnd(dummyNode.next, 2));
//        obj.dis(obj.removeNthFromEnd(dummyNode.next, 1));
//        System.out.println(obj.getIntersectionNode(dummyNode, dummyNode2).val);
//        System.out.println(obj.getIntersectionNode(dummyNode, dummyNode2).val);
//        obj.addTwoNumbers(dummyNode, dummyNode2);
//        System.out.println(obj.isPalindrome(dummyNode.next));
        ListNode[] listNodes = new ListNode[3];
        listNodes[0] = dummyNode.next;
        listNodes[1] = dummyNode2.next;
        listNodes[2] = dummyNode3.next;
//        obj.dis(obj.merge(listNodes[0], listNodes[1]));
//        obj.dis(obj.mergeKLists2(listNodes));

//        obj.dis(obj.sortList(dummyNode.next));
        ;
        ;
//        System.out.println(obj.isPalindrome(dummyNode.next));
    }
}

