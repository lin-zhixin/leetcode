package category;

import java.util.HashMap;
import java.util.Map;

class DlinkNode {
    Integer key;
    Integer val;
    DlinkNode pre;
    DlinkNode next;

    public DlinkNode() {
    }

    public DlinkNode(Integer key, Integer val, DlinkNode pre, DlinkNode next) {
        this.key = key;
        this.val = val;
        this.pre = pre;
        this.next = next;
    }
}

class LRUCache2 {
    //    双链表+map
    DlinkNode head;
    DlinkNode tail;
    int size;
    int capacity;
    Map<Integer, DlinkNode> map;

    public LRUCache2(int capacity) {
        this.capacity = capacity;
        head = new DlinkNode();
        tail = new DlinkNode();
        head.next = tail;
        tail.pre = head;
        this.size = 0;
        map = new HashMap<>();
    }

    public int get(int key) {
        if (map.containsKey(key)) {
            DlinkNode t = map.get(key);
            moveToHead(t);
            return t.val;
        }
        return -1;
    }

    public void put(int key, int value) {
        if (map.containsKey(key)) {
            DlinkNode t = map.get(key);
            t.val = value;
            moveToHead(t);
        } else {
            DlinkNode t = new DlinkNode(key, value, null, null);
            addToHead(t);
            map.put(key, t);
            size++;
            if (size > capacity) {
                map.remove(delete(tail.pre).key);
                size--;
            }
        }
        System.out.println(map);

    }

    public boolean addToHead(DlinkNode node) {
        node.next = head.next;
        node.pre = head;
        head.next.pre = node;
        head.next = node;
        return true;
    }

    public DlinkNode delete(DlinkNode node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
        return node;
    }

    public void moveToHead(DlinkNode node) {
        delete(node);
        addToHead(node);
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */