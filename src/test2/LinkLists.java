package test2;


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
    ////    86. 分隔链表
    public ListNode partition(ListNode head, int x) {
//    拆成两链表
        ListNode l = new ListNode(), res = l, r = new ListNode(), rtail = r;
        l.next = head;
        while (l.next != null) {
            if (l.next.val >= x) {
//                if (rtail == null) {
//                    rtail = new ListNode();
//                }
                rtail.next = new ListNode(l.next.val);
                rtail = rtail.next;
                l.next = l.next.next;
            } else {
                l = l.next;
            }
        }
        l.next = r.next;
        return res.next;

    }

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

    //    //234. 回文链表
    public boolean isPalindrome(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null) {

            fast = fast.next;
            if (fast != null) {
                fast = fast.next;
            }
            if (fast != null) {
                slow = slow.next;
            }
        }
        slow.next = reverse(slow.next, fast);

        ListNode l = head, r = slow.next;
        while (r != null) {
            if (l.val != r.val) {
                return false;
            }
            l = l.next;
            r = r.next;
        }
        return true;
    }

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
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode slowpre = new ListNode(), slow = head, fast = head;
        slowpre.next = head;
        left--;
        right--;
        boolean flag = false;
        while (left > 0) {
            slow = slow.next;
            slowpre = slowpre.next;
            left--;
            flag = true;
        }
        while (right > 0) {
            fast = fast.next;
            right--;
        }
        slowpre.next = reverse(slow, fast.next);
        return flag ? head : slowpre.next;
    }

    //
    public ListNode reverse(ListNode a, ListNode b) {
//        [)
        ListNode pre = b, cur = a, next;
        while (cur != b) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    // 5. Reverse Nodes in k-Group k个一组反转链表（递归）
    public ListNode reverseKGroup2(ListNode head, int k) {
        if (head == null) return null;
        ListNode slow = head, fast = head;
        int t = k;
        while ((t = t - 1) >= 0) {
            if (fast == null) {
                return head;
            }
            fast = fast.next;
        }
        ListNode newhead = reverse(slow, fast);
        slow.next = reverseKGroup2(fast, k);
        return newhead;
    }

    //    //25. Reverse Nodes in k-Group k个一组反转链表（非递归）
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode slow = new ListNode(), fast = head;
        ListNode res = slow;
        slow.next = head;
        out:
        while (fast != null) {
            for (int i = 0; i < k; i++) {
                if (fast == null) {
                    break out;
                }
                fast = fast.next;
            }
            slow.next = reverse(slow.next, fast);
            for (int i = 0; i < k && slow != null; i++) {
                slow = slow.next;
            }
        }
//        dis(res);
        return res.next;
    }

    //    25. Reverse Nodes in k-Group k个一组反转链表 递归做法
//    public ListNode reverseKGroup2(ListNode head, int k) {
//        if (head == null) {
//            return head;
//        }
//        ListNode slow = head, fast = head;
//        for (int i = 0; i < k; i++) {
//            if (fast == null) {
//                return head;
//            }
//            fast = fast.next;
//        }
//        ListNode newHead = reverse(slow, fast);
//        slow.next = reverseKGroup2(fast, k);
//        return newHead;
//    }

    //
    public void dis(ListNode head) {
        ListNode p = head;
        while (Objects.nonNull(p)) {
            System.out.print(p.val + " ");
            p = p.next;
        }
        System.out.println();
    }

    //
//
//    //19. 删除链表的倒数第 N 个结点 (双指针间隔为n+1)
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode pre = new ListNode(), res, slow = head, fast = head;
        pre.next = slow;
        res = pre;
        n--;
        while (n > 0 && fast != null) {
            fast = fast.next;
            n--;
        }
        while (fast.next != null) {
            pre = pre.next;
            slow = slow.next;
            fast = fast.next;
        }
        pre.next = pre.next.next;
        dis(res.next);
        return res.next;

    }

    //    //160. 相交链表 总共走的长度相等说明相交
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode pa = headA, pb = headB;
        while (pa != pb) {
            if (pa == null) {
                pa = headB;
            } else {
                pa = pa.next;
            }
            if (pb == null) {
                pb = headA;
            } else {
                pb = pb.next;
            }
        }
        return pa;
    }

    //    //2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode p1 = l1, p1pre = new ListNode(), p2 = l2;
        p1pre.next = p1;
        int c = 0;
        while (p1 != null && p2 != null) {
            int num = p1.val + p2.val + c;
            p1.val = num % 10;
            c = num >= 10 ? 1 : 0;
            p1 = p1.next;
            p1pre = p1pre.next;
            p2 = p2.next;
//            System.out.println("e");
        }

        while (p1 != null) {
            int num = p1.val + c;
            p1.val = num % 10;
            c = num >= 10 ? 1 : 0;
            p1 = p1.next;
            p1pre = p1pre.next;
        }
        while (p2 != null) {
            int num = p2.val + c;
            p1 = new ListNode(num % 10);
            p1pre.next = p1;
            p1pre = p1pre.next;
//            p2.val = num % 10;
            c = num >= 10 ? 1 : 0;
            p2 = p2.next;
        }
        if (c == 1) {
            p1pre.next = new ListNode(1);
        }
        dis(l1);
        return l1;
    }

    //    //237. 删除链表中的节点
    public void deleteNode(ListNode node) {
        ListNode pre = node;
        while (node.next != null) {
            int t = node.val;
            node.val = node.next.val;
            node.next.val = t;
            pre = node;
            node = node.next;
        }
        pre.next = null;
    }

    //
//    //23. 合并 K 个升序链表
    public ListNode merge(ListNode a, ListNode b) {
        ListNode p3 = new ListNode(), pa = a, pb = b, res = p3;
        while (pa != null && pb != null) {
            if (pa.val <= pb.val) {
                p3.next = new ListNode(pa.val);
                p3 = p3.next;
                pa = pa.next;
            } else {
                p3.next = new ListNode(pb.val);
                p3 = p3.next;
                pb = pb.next;
            }
        }
        while (pa != null) {
            p3.next = new ListNode(pa.val);
            p3 = p3.next;
            pa = pa.next;
        }
        while (pb != null) {
            p3.next = new ListNode(pb.val);
            p3 = p3.next;
            pb = pb.next;
        }
        return res.next;
    }

    //
    public ListNode mergeSortLinkLists(ListNode[] lists, int l, int r, int level) {
        if (l < r) {
            int mid = (l + r) / 2;
            return merge(mergeSortLinkLists(lists, l, mid, level), mergeSortLinkLists(lists, mid + 1, r, level));
        }
        return l == r ? lists[l] : null;
    }

    //
    public ListNode mergeKLists(ListNode[] lists) {
        return mergeSortLinkLists(lists, 0, lists.length - 1, 0);
    }

    //
//    //    堆做法
    public ListNode mergeKLists2(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> heap = new PriorityQueue<>((o1, o2) -> o1.val - o2.val);
        for (ListNode list : lists) {
            if (list != null)
                heap.offer(list);
        }

        ListNode p = new ListNode(), res = p;
        while (!heap.isEmpty()) {
            ListNode peek = heap.poll();
            p.next = new ListNode(peek.val);
            p = p.next;
            if (peek.next != null) {
                heap.offer(peek.next);
            }
        }
        return res.next;
    }

    //
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

    //148. 排序链表

    public ListNode sortList(ListNode head) {
        return sortList(head, null);
    }

    public ListNode sortList(ListNode head, ListNode tail) {

        if (head == null || head == tail) {
            return head;
        }
        if (head.next == tail) {
            head.next = null;
            tail.next = null;
            return merge(head, tail);
        }

        ListNode slow = head, fast = head;
        while (fast != tail) {
            slow = slow.next;
            fast = fast.next;
            if (fast != tail) {
                fast = fast.next;
            }
        }
        return merge(sortList(head, slow), sortList(slow, tail));
    }

    public ListNode merge3(ListNode a, ListNode b) {
        ListNode p3 = new ListNode(), pa = a, pb = b, res = p3;
        while (pa != b && pb != null) {
            if (pa.val <= pb.val) {
                p3.next = new ListNode(pa.val);
                p3 = p3.next;
                pa = pa.next;
            } else {
                p3.next = new ListNode(pb.val);
                p3 = p3.next;
                pb = pb.next;
            }
        }
        while (pa != null) {
            p3.next = new ListNode(pa.val);
            p3 = p3.next;
            pa = pa.next;
        }
        while (pb != null) {
            p3.next = new ListNode(pb.val);
            p3 = p3.next;
            pb = pb.next;
        }
        return res.next;
    }


    //
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

        LinkLists obj = new LinkLists();
        System.out.println();
        obj.dis(obj.reverseKGroup2(dummyNode.next, 2));
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
        obj.dis(obj.mergeKLists(listNodes));

//        obj.dis(obj.sortList(dummyNode.next));
        ;
        ;
//        System.out.println(obj.isPalindrome(dummyNode.next));
//        System.out.println(obj.reverseBetween(dummyNode.next, 1, 2));
//        System.out.println(obj.partition(dummyNode.next, 3));
    }
}

