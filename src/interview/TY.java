package interview;

import test3.Linklist;

public class TY {

    public LisNode reverse(LisNode root) {
        LisNode pre = null, p = root;
        while (p != null) {
            LisNode next = p.next;
            p.next = pre;
            pre = p;
            p = next;
        }
        return pre;
    }

    public LisNode dis(LisNode root) {
        LisNode pre = null, p = root;
        while (p != null) {
            System.out.println(p.val);
            p = p.next;
        }
        return pre;
    }

    public static void main(String[] args) {
        LisNode root = new LisNode(1, new LisNode(2, new LisNode(3, null)));
        TY ty = new TY();
        LisNode newr = ty.reverse(root);
        ty.dis(newr);

    }

}

class LisNode {
    int val;
    LisNode next;

    public LisNode() {
    }

    public LisNode(int val, LisNode next) {
        this.val = val;
        this.next = next;
    }
}
