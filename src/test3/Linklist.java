package test3;


public class Linklist {
    private ListNode f;
//    148. 排序链表 自底向上 时间 nlogn 因为自底向上的每一层涉及到的节点数都是n个，自底向上层数为logn，因此是nlogn 空间 o1 因为固定ListNode节点数量
//    自顶向下由于递归所以会带来logn的空间代价


    public ListNode sortList(ListNode head) {
        if (head == null) {
            return null;
        }
        int len = 0;
        ListNode cur = head;
        while (cur != null) {
            len++;
            cur = cur.next;
        }

        ListNode res = new ListNode(-1, head);
        for (int i = 1; i < len; i *= 2) {
            ListNode pre = res;
            cur = res.next;

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
//                    安排cur和pre：
                if (cur != null) {
                    ListNode next = cur.next;
                    cur.next = null;
                    cur = next;
                }

                pre.next = merge(m1, m2);
                while (pre.next != null) {
                    pre = pre.next;
                }
            }

        }
        return res.next;
    }


//        public ListNode sortList1(ListNode head) {
//            if (head == null) {
//                return head;
//            }
//            int length = 0;
//            ListNode node = head;
//            while (node != null) {
//                length++;
//                node = node.next;
//            }
//
//            ListNode dummyHead = new ListNode(0, head);
//            for (int subLength = 1; subLength < length; subLength <<= 1) {
//                ListNode prev = dummyHead, curr = dummyHead.next;
////                一次归并
//                while (curr != null) {
//                    ListNode head1 = curr;
//                    for (int i = 1; i < subLength && curr.next != null; i++) {
//                        curr = curr.next;
//                    }
//                    ListNode head2 = curr.next;
//                    curr.next = null;
//                    curr = head2;
//                    for (int i = 1; i < subLength && curr != null && curr.next != null; i++) {
//                        curr = curr.next;
//                    }
//
//                    ListNode next = null;
//                    if (curr != null) {
//                        next = curr.next;
//                        curr.next = null;
//                    }
//                    ListNode merged = merge(head1, head2);
//                    prev.next = merged;
//                    while (prev.next != null) {
//                        prev = prev.next;
//                    }
//                    curr = next;
//                }
//            }
//            return dummyHead.next;
//        }

    public ListNode merge(ListNode head1, ListNode head2) {
        ListNode t1 = head1, t2 = head2, res = new ListNode(), t3 = res;
        while (t1 != null && t2 != null) {
            if (t1.val <= t2.val) {
                t3.next = t1;
                t1 = t1.next;
            } else {
                t3.next = t2;
                t2 = t2.next;
            }
            t3 = t3.next;
        }
        if (t1 != null) {
            t3.next = t1;

        } else if (t2 != null) {
            t3.next = t2;
        }
        return res.next;
    }


    //19. 删除链表的倒数第 N 个结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) {
            return null;
        }

        ListNode pre = null, slow = head, fast = head;
        while ((n = n - 1) > 0) {
            System.out.println(n);
            fast = fast.next;
        }
        if (fast == null || fast.next == null) {
            head = head.next;
            return head;
        }
        while (fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next;
        }
        pre.next = slow.next;
        return head;

    }

    //143. 重排链表
    public void reorderList(ListNode head) {
        if (head == null) return;
        ListNode pre = null, s = head, f = head;
        while (f != null) {
            pre = s;
            s = s.next;
            f = f.next;
            if (f != null) {
                f = f.next;
            }
        }
        pre.next = null;
        s = reverse(s);
//        System.out.println("re s:");
//        dis(s);
//        System.out.println(" head:");
//        dis(head);
        head = merge3(head, s);
//        dis(head);
    }

    public ListNode reverse(ListNode head) {
        ListNode pre = null, p = head;
        while (p != null) {
            ListNode next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        return pre;
    }

    public ListNode merge3(ListNode a, ListNode b) {
        ListNode newhead = new ListNode(-1), pa = a, pb = b, pnh = newhead;
        while (pa != null || pb != null) {
            if (pa != null) {
                ListNode next = pa.next;
                pnh.next = pa;
                pa.next = null;
                pa = next;
                pnh = pnh.next;
            }
            if (pb != null) {
                ListNode next = pb.next;
                pnh.next = pb;
                pb.next = null;
                pb = next;
                pnh = pnh.next;
            }

        }
        return newhead.next;
    }

    //92. 反转链表 II
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (head == null) return null;
        ListNode pre = null, s = head, f = head;
        int n = right - left;
        while ((n = n - 1) >= 0) {
            f = f.next;
        }
        while ((left = left - 1) > 0) {
            pre = s;
            s = s.next;
            f = f.next;
        }
        if (pre==null){
            return reverse2(s, f);
        }else {
            pre.next = reverse2(s, f);
        }
        return head;
    }

    public ListNode reverse2(ListNode l, ListNode r) {
        ListNode pre = r.next, p = l, end = r.next;
        while (p != end) {
            ListNode next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        return pre;
    }

}

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

